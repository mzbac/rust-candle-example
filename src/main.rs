use anyhow::{Error as E, Result};
use candle_core::Device;
use candle_transformers::models::mistral::Config;
use candle_transformers::models::quantized_mistral::Model as QMistral;

use std::path::PathBuf;
use tokenizers::Tokenizer;

mod model;
use model::TextGeneration;

fn main() -> Result<()> {
    let tokenizer_filename = PathBuf::from("./models/tokenizer.json");
    let model_name = PathBuf::from("./models/model-q4k.gguf");
    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

    let config = Config::config_7b_v0_1(true);
    let vb = candle_transformers::quantized_var_builder::VarBuilder::from_gguf(model_name)?;
    let model = QMistral::new(&config, vb)?;
    let device = Device::Cpu;

    let seed = 299792458;
    let temperature = Some(0.7);
    let top_p = Some(0.95);
    let repeat_penalty = 1.1;
    let repeat_last_n = 64;
    let sample_len = 400;

    let mut pipeline = TextGeneration::new(
        model,
        tokenizer,
        seed,
        temperature,
        top_p,
        repeat_penalty,
        repeat_last_n,
        &device,
    );
    let res = pipeline.run("hello world".into(), sample_len)?;
    println!("result: {:?}", res.join(""));
    Ok(())
}
