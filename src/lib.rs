mod inference;

use anyhow::{anyhow, Error, Result};
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use hf_hub::{api::sync::Api, Repo, RepoType};
use inference::models::{BertConfig, BertModel, RobertaConfig, RobertaModel, DTYPE};
pub use inference::{Model, ModelConfig, ModelInput, WeigthBackend};
use serde_json::Value;
use std::path::PathBuf;
use tokenizers::{PaddingParams, Tokenizer};

pub struct PredictorConfig {
    pub tokenizer_config: PathBuf,
    pub model_weights: WeigthBackend,
    pub model_config: ModelConfig,
}

impl PredictorConfig {
    pub fn load(model_id: Option<String>, revision: Option<String>) -> Result<Self> {
        let default_model = "intfloat/multilingual-e5-base".to_string();
        let default_revision = "main".to_string();

        let (model_id, revision) = match (model_id.to_owned(), revision.to_owned()) {
            (Some(model_id), Some(revision)) => (model_id, revision),
            (Some(model_id), None) => (model_id, "main".to_string()),
            (None, Some(revision)) => (default_model, revision),
            (None, None) => (default_model, default_revision),
        };

        let repo = Repo::with_revision(model_id, RepoType::Model, revision);
        let (config_filename, tokenizer_config, model_weights) = {
            let api = Api::new()?;
            let api = api.repo(repo);
            let config = api.get("config.json")?;
            let tokenizer = api.get("tokenizer.json")?;
            let weights = match api.get("pytorch_model.bin") {
                Ok(weights) => WeigthBackend::PyTorch(weights),
                Err(err) => {
                    WeigthBackend::SafeTensors(api.get("model.safetensors")?) // TODO add error handling
                }
            };
            (config, tokenizer, weights)
        };

        let config = std::fs::read_to_string(config_filename)?;

        let model_config = match get_model_type(&config).as_deref() {
            Some("bert") => ModelConfig::Bert(serde_json::from_str::<BertConfig>(&config)?),
            Some("xlm-roberta" | "roberta") => {
                ModelConfig::Roberta(serde_json::from_str::<RobertaConfig>(&config)?)
            }
            Some(model_type) => return Err(anyhow!("Model {:?} is not supported", model_type)),
            None => return Err(anyhow!("No model_type found in model config")),
        };

        Ok(Self {
            tokenizer_config,
            model_weights,
            model_config,
        })
    }
}
pub struct Predictor {
    pub tokenizer: Tokenizer,
    pub model: Box<dyn Model + Send>,
}

impl Predictor {
    pub fn load(config: &PredictorConfig) -> Result<Self> {
        let mut tokenizer = Tokenizer::from_file(&config.tokenizer_config).map_err(Error::msg)?;
        // TODO: add proper tokenizer config
        // some questionable implementaion of parameters access
        let tokenizer = tokenizer
            .with_padding(None)
            .with_truncation(None)
            .map_err(Error::msg)?;
        let tokenizer = Tokenizer::from(tokenizer.clone());

        let vb = match &config.model_weights {
            WeigthBackend::PyTorch(path) => VarBuilder::from_pth(&path, DTYPE, &Device::Cpu)?,
            WeigthBackend::SafeTensors(path) => unsafe {
                VarBuilder::from_mmaped_safetensors(&[path], DTYPE, &Device::Cpu)?
            },
        };

        let model: Box<dyn Model + Send> = match &config.model_config {
            ModelConfig::Bert(config) => Box::new(BertModel::load(vb, &config)?),
            ModelConfig::Roberta(config) => Box::new(RobertaModel::load(vb, &config)?),
        };

        Ok(Self { tokenizer, model })
    }

    pub fn process_single(&self, input_text: String) -> Result<Tensor> {
        let tokens = self
            .tokenizer
            .encode(input_text, true)
            .map_err(Error::msg)?
            .get_ids()
            .to_vec();
        let token_ids = Tensor::new(&tokens[..], &Device::Cpu)?.unsqueeze(0)?;
        let token_type_ids = token_ids.zeros_like()?;

        let ys = self.model.embed_single(ModelInput {
            token_ids,
            token_type_ids,
        })?;

        // apply mean pooling
        let (_n_sentence, n_tokens, _hidden_size) = ys.dims3()?;
        let embeddings = (ys.sum(1)? / (n_tokens as f64))?;
        normalize_l2(&embeddings)
    }
}

fn get_model_type(config_str: &String) -> Option<String> {
    // TODO: add error handling
    let config_json: Value =
        serde_json::from_str(config_str).expect("problem parsing model config.json");
    config_json
        .get("model_type")
        .and_then(|value| value.as_str())
        .map(str::to_string)
}

pub fn normalize_l2(v: &Tensor) -> Result<Tensor> {
    Ok(v.broadcast_div(&v.sqr()?.sum_keepdim(1)?.sqrt()?)?)
}
