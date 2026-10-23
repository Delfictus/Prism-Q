//! Operand schema surface for compiler-emitted PrismCode instructions.

use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Operand {
    pub name: String,
    pub role: String,
    pub value: Value,
}
