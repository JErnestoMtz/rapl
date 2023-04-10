#[derive(Debug)]
pub struct DimError {
    details: String,
}

impl DimError {
    pub fn new(msg: &str) -> DimError {
        DimError {
            details: msg.to_string(),
        }
    }
}
