pub mod models;

use candle_core::{Result, Tensor};
use models::{BertConfig, BertModel, RobertaConfig, RobertaModel};
use std::path::PathBuf;

#[derive(Debug)]
pub struct ModelInput {
    pub token_ids: Tensor,
    pub token_type_ids: Tensor,
}

#[derive(Debug)]
pub struct Batch {
    pub input_ids: Vec<u32>,
    pub token_type_ids: Vec<u32>,
    pub position_ids: Vec<u32>,
    pub cumulative_seq_lengths: Vec<u32>,
    pub max_length: u32,
}

pub enum WeigthBackend {
    PyTorch(PathBuf),
    SafeTensors(PathBuf),
}
pub enum ModelConfig {
    Bert(BertConfig),
    Roberta(RobertaConfig),
}

pub trait Model {
    fn embed_single(&self, _input: ModelInput) -> Result<Tensor> {
        candle_core::bail!("`embed_single` is not implemented for this model");
    }

    fn embed(&self, _batch: Batch) -> Result<Tensor> {
        candle_core::bail!("`embed` is not implemented for this model");
    }

    fn predict(&self, _batch: Batch) -> Result<Tensor> {
        candle_core::bail!("`predict is not implemented for this model");
    }
}

impl Model for BertModel {
    fn embed_single(&self, input: ModelInput) -> candle_core::Result<candle_core::Tensor> {
        self.forward(&input.token_ids, &input.token_type_ids)
    }
}

impl Model for RobertaModel {
    fn embed_single(&self, input: ModelInput) -> candle_core::Result<candle_core::Tensor> {
        self.forward(&input.token_ids, &input.token_type_ids)
    }
}
