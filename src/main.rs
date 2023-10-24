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
    let filenames = vec![PathBuf::from("./models/model-q4k.gguf")];
    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

    let config = Config::config_7b_v0_1(true);
    let filename = &filenames[0];
    let vb = candle_transformers::quantized_var_builder::VarBuilder::from_gguf(filename)?;
    let model = QMistral::new(&config, vb)?;
    let device = Device::Cpu;
    let seed = 299792458;

    let mut pipeline = TextGeneration::new(
        model,
        tokenizer,
        seed,
        Some(0.7),
        Some(0.95),
        1.1,
        64,
        &device,
    );
    let res = pipeline.run("hello world".into(), 400)?;
    println!("result: {:?}", res.join(""));
    Ok(())
}
