use actix_web::{web, App, HttpServer, Responder};
use std::sync::{Arc, Mutex};
use inference_server::{Predictor, PredictorConfig};
use anyhow::Result;

#[derive(Clone)]
struct AppState {
    predictor: Arc<Mutex<Predictor>>,
}

async fn say_hi() -> impl Responder {
    format!("hi there")
}

async fn embedd(data: web::Data<AppState>) -> impl Responder {
    let input = "test string".to_string();
    let predictor = data.predictor.lock().unwrap();
    let embeddings = predictor.process_single(input).unwrap();

    format!("embeddings for \"test string\": {}", embeddings)
}

#[actix_web::main]
async fn main() -> Result<()> {
    let model_id = Some("intfloat/multilingual-e5-base".to_string());
    let revision = Some("main".to_string());
    let model_config = PredictorConfig::load(model_id, revision)?;
    let predictor = Predictor::load(&model_config)?;
    println!("completed model load");

    let data = AppState {
        predictor: Arc::new(Mutex::from(predictor)),
    };
    
    HttpServer::new(move || {
        App::new()
            .app_data(web::Data::new(data.clone()))
            .route("/", web::to(say_hi))
            .route("/embedd", web::to(embedd))
    })
    .bind(("127.0.0.1", 8080))?
    .run()
    .await.map_err(anyhow::Error::from)
}
