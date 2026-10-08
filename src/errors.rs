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

impl std::fmt::Display for DimError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.details)
    }
}

impl std::error::Error for DimError {}
