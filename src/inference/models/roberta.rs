use std::collections::HashMap;

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{embedding, layer_norm, linear, Activation, Dropout, LayerNorm, Linear};
use candle_nn::{Embedding, Module, VarBuilder};

use serde::Deserialize;

pub const FLOATING_DTYPE: DType = DType::F32;
pub const LONG_DTYPE: DType = DType::I64;

pub const TRAIN: bool = false;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
enum PositionEmbeddingType {
    #[default]
    Absolute,
}
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct RobertaConfig {
    vocab_size: usize,
    hidden_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    intermediate_size: usize,
    hidden_act: Activation,
    hidden_dropout_prob: f64,
    max_position_embeddings: usize,
    type_vocab_size: usize,
    initializer_range: f64,
    layer_norm_eps: f64,
    pad_token_id: usize,
    bos_token_id: usize,
    eos_token_id: usize,
    #[serde(default)]
    position_embedding_type: PositionEmbeddingType,
    #[serde(default)]
    use_cache: bool,
    classifier_dropout: Option<f64>,
    model_type: Option<String>,
    problem_type: Option<String>,
    _num_labels: Option<usize>,
    id2label: Option<HashMap<String, String>>,
    label2id: Option<HashMap<String, usize>>,
}

impl Default for RobertaConfig {
    fn default() -> Self {
        Self {
            vocab_size: 50265,
            hidden_size: 768,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            intermediate_size: 3072,
            hidden_act: Activation::Gelu,
            hidden_dropout_prob: 0.1,
            max_position_embeddings: 512,
            type_vocab_size: 2,
            initializer_range: 0.02,
            layer_norm_eps: 1e-12,
            pad_token_id: 1,
            bos_token_id: 0,
            eos_token_id: 2,
            position_embedding_type: PositionEmbeddingType::Absolute,
            use_cache: true,
            classifier_dropout: None,
            model_type: Some("roberta".to_string()),
            problem_type: None,
            _num_labels: Some(3),
            id2label: None,
            label2id: None,
        }
    }
}

fn cumsum_2d(mask: &Tensor, dim: u8, device: &Device) -> Result<Tensor> {
    let mask = mask.to_vec2::<u8>()?;

    let rows = mask.len();
    let cols = mask[0].len();

    let mut result = mask.clone();

    match dim {
        0 => {
            // Cumulative sum along rows
            for i in 0..rows {
                for j in 1..cols {
                    result[i][j] += result[i][j - 1];
                }
            }
        }
        1 => {
            // Cumulative sum along columns
            for j in 0..cols {
                for i in 1..rows {
                    result[i][j] += result[i - 1][j];
                }
            }
        }
        _ => panic!("Dimension not supported"),
    }

    let result = Tensor::new(result, &device)?;

    Ok(result)
}

pub fn create_position_ids_from_input_ids(
    input_ids: &Tensor,
    padding_idx: u32,
    past_key_values_length: u8,
) -> Result<Tensor> {
    let mask = input_ids.ne(padding_idx)?;
    let incremental_indices = cumsum_2d(&mask, 0, input_ids.device())?;

    let incremental_indices = incremental_indices
        .broadcast_add(&Tensor::new(&[past_key_values_length], input_ids.device())?)?;

    Ok(incremental_indices)
}

pub struct RobertaEmbeddings {
    word_embeddings: Embedding,
    position_embeddings: Option<Embedding>,
    token_type_embeddings: Embedding,
    layer_norm: LayerNorm,
    dropout: Dropout,
    pub padding_idx: u32,
}

impl RobertaEmbeddings {
    pub fn load(vb: VarBuilder, config: &RobertaConfig) -> Result<Self> {
        let word_embeddings = embedding(
            config.vocab_size,
            config.hidden_size,
            vb.pp("word_embeddings"),
        )?;
        let position_embeddings = embedding(
            config.max_position_embeddings,
            config.hidden_size,
            vb.pp("position_embeddings"),
        )?;
        let token_type_embeddings = embedding(
            config.type_vocab_size,
            config.hidden_size,
            vb.pp("token_type_embeddings"),
        )?;
        let layer_norm = layer_norm(
            config.hidden_size,
            config.layer_norm_eps,
            vb.pp("LayerNorm"),
        )?;
        let padding_idx = config.pad_token_id as u32;

        Ok(Self {
            word_embeddings,
            position_embeddings: Some(position_embeddings),
            token_type_embeddings,
            layer_norm,
            dropout: Dropout::new(config.hidden_dropout_prob as f32),
            padding_idx,
        })
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        token_type_ids: &Tensor,
        position_ids: Option<&Tensor>,
        inputs_embeds: Option<&Tensor>,
    ) -> Result<Tensor> {
        let position_ids = match position_ids {
            Some(ids) => ids.to_owned(),
            None => {
                if Option::is_some(&inputs_embeds) {
                    let position_ids =
                        self.create_position_ids_from_input_embeds(inputs_embeds.unwrap())?;
                    position_ids
                } else {
                    let position_ids =
                        create_position_ids_from_input_ids(input_ids, self.padding_idx, 1)?;
                    position_ids
                }
            }
        };

        let inputs_embeds: Tensor = match inputs_embeds {
            Some(embeds) => embeds.to_owned(),
            None => {
                let embeds = self.word_embeddings.forward(input_ids)?;
                embeds
            }
        };

        let token_type_embeddings = self.token_type_embeddings.forward(token_type_ids)?;
        let mut embeddings = (inputs_embeds + token_type_embeddings)?;

        if let Some(position_embeddings) = &self.position_embeddings {
            embeddings = embeddings.broadcast_add(&position_embeddings.forward(&position_ids)?)?
        }

        let embeddings = self.layer_norm.forward(&embeddings)?;
        let embeddings = self.dropout.forward(&embeddings, TRAIN)?;

        Ok(embeddings)
    }

    pub fn create_position_ids_from_input_embeds(&self, input_embeds: &Tensor) -> Result<Tensor> {
        let input_shape = input_embeds.dims3()?;
        let seq_length = input_shape.1;

        println!("seq_length: {:?}", seq_length);
        let mut position_ids = Tensor::arange(
            self.padding_idx + 1,
            seq_length as u32 + self.padding_idx + 1,
            &Device::Cpu,
        )?;

        println!("position_ids: {:?}", position_ids);

        position_ids = position_ids
            .unsqueeze(0)?
            .expand((input_shape.0, input_shape.1))?;
        Ok(position_ids)
    }
}

struct RobertaSelfAttention {
    query: Linear,
    key: Linear,
    value: Linear,
    dropout: Dropout,
    num_attention_heads: usize,
    attention_head_size: usize,
}

impl RobertaSelfAttention {
    fn load(vb: VarBuilder, config: &RobertaConfig) -> Result<Self> {
        let attention_head_size = config.hidden_size / config.num_attention_heads;
        let all_head_size = config.num_attention_heads * attention_head_size;
        let dropout = Dropout::new(config.hidden_dropout_prob as f32);
        let hidden_size = config.hidden_size;
        let query = linear(hidden_size, all_head_size, vb.pp("query"))?;
        let value = linear(hidden_size, all_head_size, vb.pp("value"))?;
        let key = linear(hidden_size, all_head_size, vb.pp("key"))?;
        Ok(Self {
            query,
            key,
            value,
            dropout,
            num_attention_heads: config.num_attention_heads,
            attention_head_size,
        })
    }

    fn transpose_for_scores(&self, xs: &Tensor) -> Result<Tensor> {
        let mut new_x_shape = xs.dims().to_vec();
        new_x_shape.pop();
        new_x_shape.push(self.num_attention_heads);
        new_x_shape.push(self.attention_head_size);
        let xs = xs.reshape(new_x_shape.as_slice())?.transpose(1, 2)?;
        xs.contiguous()
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let query_layer = self.query.forward(hidden_states)?;
        let key_layer = self.key.forward(hidden_states)?;
        let value_layer = self.value.forward(hidden_states)?;

        let query_layer = self.transpose_for_scores(&query_layer)?;
        let key_layer = self.transpose_for_scores(&key_layer)?;
        let value_layer = self.transpose_for_scores(&value_layer)?;

        let attention_scores = query_layer.matmul(&key_layer.t()?)?;
        let attention_scores = (attention_scores / (self.attention_head_size as f64).sqrt())?;
        let attention_probs =
            { candle_nn::ops::softmax(&attention_scores, candle_core::D::Minus1)? };
        let attention_probs = self.dropout.forward(&attention_probs, TRAIN)?;

        let context_layer = attention_probs.matmul(&value_layer)?;
        let context_layer = context_layer.transpose(1, 2)?.contiguous()?;
        let context_layer = context_layer.flatten_from(candle_core::D::Minus2)?;
        Ok(context_layer)
    }
}

struct RobertaSelfOutput {
    dense: Linear,
    layer_norm: LayerNorm,
    dropout: Dropout,
}

impl RobertaSelfOutput {
    fn load(vb: VarBuilder, config: &RobertaConfig) -> Result<Self> {
        let dense = linear(config.hidden_size, config.hidden_size, vb.pp("dense"))?;
        let layer_norm = layer_norm(
            config.hidden_size,
            config.layer_norm_eps,
            vb.pp("LayerNorm"),
        )?;
        let dropout = Dropout::new(config.hidden_dropout_prob as f32);
        Ok(Self {
            dense,
            layer_norm,
            dropout,
        })
    }

    fn forward(&self, hidden_states: &Tensor, input_tensor: &Tensor) -> Result<Tensor> {
        let hidden_states = self.dense.forward(hidden_states)?;
        let hidden_states = self.dropout.forward(&hidden_states, TRAIN)?;
        self.layer_norm.forward(&(hidden_states + input_tensor)?)
    }
}

struct RobertaAttention {
    self_attention: RobertaSelfAttention,
    self_output: RobertaSelfOutput,
}

impl RobertaAttention {
    fn load(vb: VarBuilder, config: &RobertaConfig) -> Result<Self> {
        let self_attention = RobertaSelfAttention::load(vb.pp("self"), config)?;
        let self_output = RobertaSelfOutput::load(vb.pp("output"), config)?;
        Ok(Self {
            self_attention,
            self_output,
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let self_outputs = self.self_attention.forward(hidden_states)?;
        let attention_output = self.self_output.forward(&self_outputs, hidden_states)?;
        Ok(attention_output)
    }
}

struct HiddenActLayer {
    act: Activation,
    // TODO add tracing to whole class
    // span: tracing::Span,
}

impl HiddenActLayer {
    fn new(act: Activation) -> Self {
        // TODO add tracing
        // let span = tracing::span!(tracing::Level::TRACE, "hidden-act");
        Self { act }
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // let _enter = self.span.enter();
        // match self.act {
        //     // https://github.com/huggingface/transformers/blob/cd4584e3c809bb9e1392ccd3fe38b40daba5519a/src/transformers/activations.py#L213
        //     Activation::Gelu => xs.gelu_erf(),
        //     Activation::Relu => xs.relu(),
        // }
        self.act.forward(xs)
    }
}
struct RobertaIntermediate {
    dense: Linear,
    intermediate_act: HiddenActLayer,
}

impl RobertaIntermediate {
    fn load(vb: VarBuilder, config: &RobertaConfig) -> Result<Self> {
        let dense = linear(config.hidden_size, config.intermediate_size, vb.pp("dense"))?;
        Ok(Self {
            dense,
            intermediate_act: HiddenActLayer::new(config.hidden_act),
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let hidden_states = self.dense.forward(hidden_states)?;
        let ys = self.intermediate_act.forward(&hidden_states)?;
        Ok(ys)
    }
}

struct RobertaOutput {
    dense: Linear,
    layer_norm: LayerNorm,
    dropout: Dropout,
}

impl RobertaOutput {
    fn load(vb: VarBuilder, config: &RobertaConfig) -> Result<Self> {
        let dense = linear(config.intermediate_size, config.hidden_size, vb.pp("dense"))?;
        let layer_norm = layer_norm(
            config.hidden_size,
            config.layer_norm_eps,
            vb.pp("LayerNorm"),
        )?;
        let dropout = Dropout::new(config.hidden_dropout_prob as f32);
        Ok(Self {
            dense,
            layer_norm,
            dropout,
        })
    }

    fn forward(&self, hidden_states: &Tensor, input_tensor: &Tensor) -> Result<Tensor> {
        let hidden_states = self.dense.forward(hidden_states)?;
        let hidden_states = self.dropout.forward(&hidden_states, TRAIN)?;
        self.layer_norm.forward(&(hidden_states + input_tensor)?)
    }
}

struct RobertaLayer {
    attention: RobertaAttention,
    intermediate: RobertaIntermediate,
    output: RobertaOutput,
}

impl RobertaLayer {
    fn load(vb: VarBuilder, config: &RobertaConfig) -> Result<Self> {
        let attention = RobertaAttention::load(vb.pp("attention"), config)?;
        let intermediate = RobertaIntermediate::load(vb.pp("intermediate"), config)?;
        let output = RobertaOutput::load(vb.pp("output"), config)?;
        Ok(Self {
            attention,
            intermediate,
            output,
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let attention_output = self.attention.forward(hidden_states)?;

        let intermediate_output = self.intermediate.forward(&attention_output)?;
        let layer_output = self
            .output
            .forward(&intermediate_output, &attention_output)?;
        Ok(layer_output)
    }
}

struct RobertaEncoder {
    layers: Vec<RobertaLayer>,
}

impl RobertaEncoder {
    fn load(vb: VarBuilder, config: &RobertaConfig) -> Result<Self> {
        let layers = (0..config.num_hidden_layers)
            .map(|index| RobertaLayer::load(vb.pp(&format!("layer.{index}")), config))
            .collect::<Result<Vec<_>>>()?;
        Ok(RobertaEncoder { layers })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let mut hidden_states = hidden_states.clone();
        for layer in self.layers.iter() {
            hidden_states = layer.forward(&hidden_states)?
        }
        Ok(hidden_states)
    }
}

pub struct RobertaModel {
    embeddings: RobertaEmbeddings,
    encoder: RobertaEncoder,
    pub device: Device,
}

impl RobertaModel {
    pub fn load(vb: VarBuilder, config: &RobertaConfig) -> Result<Self> {
        let (embeddings, encoder) = match (
            RobertaEmbeddings::load(vb.pp("embeddings"), config),
            RobertaEncoder::load(vb.pp("encoder"), config),
        ) {
            (Ok(embeddings), Ok(encoder)) => (embeddings, encoder),
            (Err(err), _) | (_, Err(err)) => {
                if let Some(model_type) = &config.model_type {
                    if let (Ok(embeddings), Ok(encoder)) = (
                        RobertaEmbeddings::load(vb.pp(&format!("{model_type}.embeddings")), config),
                        RobertaEncoder::load(vb.pp(&format!("{model_type}.encoder")), config),
                    ) {
                        (embeddings, encoder)
                    } else {
                        return Err(err);
                    }
                } else {
                    return Err(err);
                }
            }
        };
        Ok(Self {
            embeddings,
            encoder,
            device: vb.device().clone(),
        })
    }

    pub fn forward(&self, input_ids: &Tensor, token_type_ids: &Tensor) -> Result<Tensor> {
        let embedding_output = self
            .embeddings
            .forward(input_ids, token_type_ids, None, None)?;
        let sequence_output = self.encoder.forward(&embedding_output)?;
        Ok(sequence_output)
    }
}
