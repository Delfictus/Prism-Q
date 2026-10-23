//! Enforced compiler-owned operand contracts. This is an explicit, bounded
//! schema vocabulary; an unknown constraint is an error, never an annotation.
use serde_json::Value;
use std::cmp::Ordering;
use std::collections::BTreeSet;

const MAX_DEPTH: usize = 64;
const TYPES: &[&str] = &["null", "boolean", "object", "array", "number", "integer", "string"];

pub(crate) fn check_schema(schema: &Value) -> Result<(), String> {
    check_schema_at(schema, 0)
}

fn check_schema_at(schema: &Value, depth: usize) -> Result<(), String> {
    if depth > MAX_DEPTH { return Err("operand schema nesting exceeds 64".into()); }
    // Null is the historical default for an opcode with no declared contract.
    if schema.is_null() || schema.is_boolean() { return Ok(()); }
    let object = schema.as_object().ok_or("operand schema must be an object, boolean, or null")?;
    for (key, value) in object {
        match key.as_str() {
            "title" | "description" | "$comment" => {
                if !value.is_string() { return Err(format!("{key} must be a string")); }
            }
            "default" => {}
            "examples" => { if !value.is_array() { return Err("examples must be an array".into()); } }
            "type" => {
                let valid = |v: &Value| v.as_str().is_some_and(|s| TYPES.contains(&s));
                if !valid(value) && !value.as_array().is_some_and(|a| !a.is_empty() && a.iter().all(valid)) {
                    return Err("type must name a supported type or a nonempty type array".into());
                }
            }
            "required" => {
                let rows = value.as_array().ok_or("required must be an array")?;
                let mut names = BTreeSet::new();
                for name in rows {
                    let name = name.as_str().ok_or("required entries must be strings")?;
                    if !names.insert(name) { return Err("required entries must be unique".into()); }
                }
            }
            "properties" => {
                for child in value.as_object().ok_or("properties must be an object")?.values() {
                    check_schema_at(child, depth + 1)?;
                }
            }
            "additionalProperties" | "items" | "not" => {
                // Nested null is permitted as the same historical unrestricted contract.
                check_schema_at(value, depth + 1)?;
            }
            "prefixItems" | "allOf" | "anyOf" | "oneOf" => {
                let rows = value.as_array().ok_or_else(|| format!("{key} must be an array"))?;
                if rows.is_empty() && key != "prefixItems" { return Err(format!("{key} must be nonempty")); }
                for child in rows { check_schema_at(child, depth + 1)?; }
            }
            "enum" => {
                let rows = value.as_array().ok_or("enum must be an array")?;
                if rows.is_empty() { return Err("enum must be nonempty".into()); }
                for (index, row) in rows.iter().enumerate() {
                    if rows[..index].iter().any(|prior| equivalent(prior, row)) { return Err("enum entries must be unique".into()); }
                }
            }
            "const" => {}
            "minimum" | "maximum" | "exclusiveMinimum" | "exclusiveMaximum" => {
                if !value.is_number() { return Err(format!("{key} must be numeric")); }
            }
            "minLength" | "maxLength" | "minItems" | "maxItems" | "minProperties" | "maxProperties" => {
                if value.as_u64().is_none() { return Err(format!("{key} must be a nonnegative integer")); }
            }
            "uniqueItems" => { if !value.is_boolean() { return Err("uniqueItems must be boolean".into()); } }
            other => return Err(format!("unsupported operand schema constraint: {other}")),
        }
    }
    Ok(())
}

fn numeric_order(a: &Value, b: &Value) -> Ordering {
    // Compare integer JSON representations exactly, including against a float
    // bound. Converting a large integer to f64 first could erase a violation.
    fn integer(v: &Value) -> Option<i128> {
        v.as_i64().map(i128::from).or_else(|| v.as_u64().map(i128::from))
    }
    fn integer_float(i: i128, f: f64) -> Ordering {
        if f < -9223372036854775808.0 { return Ordering::Greater; }
        if f >= 18446744073709551616.0 { return Ordering::Less; }
        match i.cmp(&(f as i128)) {
            Ordering::Equal if f.fract() > 0.0 => Ordering::Less,
            Ordering::Equal if f.fract() < 0.0 => Ordering::Greater,
            order => order,
        }
    }
    match (integer(a), integer(b)) {
        (Some(a), Some(b)) => a.cmp(&b),
        (Some(a), None) => integer_float(a, b.as_f64().unwrap()),
        (None, Some(b)) => integer_float(b, a.as_f64().unwrap()).reverse(),
        (None, None) => a.as_f64().unwrap().partial_cmp(&b.as_f64().unwrap()).unwrap(),
    }
}

// Numeric equality follows numeric value, including within objects and arrays:
// 1 and 1.0 are the same constraint value, while adjacent large integers remain
// distinct. This applies consistently to enum, const, and uniqueItems.
fn equivalent(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Number(_), Value::Number(_)) => numeric_order(a, b) == Ordering::Equal,
        (Value::Array(a), Value::Array(b)) => a.len() == b.len()
            && a.iter().zip(b).all(|(a,b)| equivalent(a,b)),
        (Value::Object(a), Value::Object(b)) => a.len() == b.len()
            && a.iter().all(|(key,a)| b.get(key).is_some_and(|b| equivalent(a,b))),
        _ => a == b,
    }
}

fn matches_type(value: &Value, kind: &str) -> bool {
    match kind {
        "null" => value.is_null(), "boolean" => value.is_boolean(),
        "object" => value.is_object(), "array" => value.is_array(),
        "number" => value.is_number(),
        "integer" => value.as_i64().is_some() || value.as_u64().is_some()
            || value.as_f64().is_some_and(|v| v.fract() == 0.0),
        "string" => value.is_string(), _ => false,
    }
}

pub(crate) fn check_value(schema: &Value, value: &Value) -> Result<(), String> {
    check_at(schema, value, "operands", 0)
}

fn check_at(schema: &Value, value: &Value, path: &str, depth: usize) -> Result<(), String> {
    if depth > MAX_DEPTH { return Err(format!("{path}: nesting exceeds 64")); }
    if schema == &Value::Bool(false) { return Err(format!("{path}: forbidden by schema")); }
    let Some(rules) = schema.as_object() else { return Ok(()); };
    if let Some(kind) = rules.get("type") {
        let pass = kind.as_str().is_some_and(|s| matches_type(value,s))
            || kind.as_array().is_some_and(|a| a.iter().any(|v| matches_type(value,v.as_str().unwrap())));
        if !pass { return Err(format!("{path}: type mismatch")); }
    }
    if let Some(expected) = rules.get("const") {
        if !equivalent(expected, value) { return Err(format!("{path}: const mismatch")); }
    }
    if let Some(options) = rules.get("enum") {
        if !options.as_array().unwrap().iter().any(|option| equivalent(option, value)) { return Err(format!("{path}: value outside enum")); }
    }
    for combinator in ["allOf", "anyOf", "oneOf"] {
        if let Some(children) = rules.get(combinator) {
            let children = children.as_array().unwrap();
            let passes = children.iter().filter(|s| check_at(s,value,path,depth+1).is_ok()).count();
            if match combinator { "allOf" => passes != children.len(), "anyOf" => passes == 0, _ => passes != 1 } {
                return Err(format!("{path}: {combinator} failed"));
            }
        }
    }
    if rules.get("not").is_some_and(|s| check_at(s,value,path,depth+1).is_ok()) {
        return Err(format!("{path}: not constraint failed"));
    }
    if value.is_number() {
        for key in ["minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"] {
            if let Some(bound) = rules.get(key) {
                let order = numeric_order(value,bound);
                let fail = match key { "minimum" => order == Ordering::Less,
                    "maximum" => order == Ordering::Greater,
                    "exclusiveMinimum" => order != Ordering::Greater, _ => order != Ordering::Less };
                if fail { return Err(format!("{path}: {key} violated")); }
            }
        }
    }
    let length = if let Some(s) = value.as_str() { Some((s.chars().count(), "minLength", "maxLength")) }
        else if let Some(a) = value.as_array() { Some((a.len(), "minItems", "maxItems")) }
        else { value.as_object().map(|o| (o.len(), "minProperties", "maxProperties")) };
    if let Some((length, min, max)) = length {
        if rules.get(min).is_some_and(|n| (length as u64) < n.as_u64().unwrap())
            || rules.get(max).is_some_and(|n| (length as u64) > n.as_u64().unwrap()) {
            return Err(format!("{path}: length/count bound violated"));
        }
    }
    if let Some(object) = value.as_object() {
        if let Some(required) = rules.get("required") {
            for name in required.as_array().unwrap() {
                if !object.contains_key(name.as_str().unwrap()) { return Err(format!("{path}: required operand {name} missing")); }
            }
        }
        let properties = rules.get("properties").and_then(Value::as_object);
        for (name, child) in object {
            let rule = properties.and_then(|p| p.get(name)).or_else(|| rules.get("additionalProperties"));
            if let Some(rule) = rule { check_at(rule,child,&format!("{path}.{name}"),depth+1)?; }
        }
    }
    if let Some(array) = value.as_array() {
        if rules.get("uniqueItems") == Some(&Value::Bool(true)) {
            for (i, child) in array.iter().enumerate() {
                if array[..i].iter().any(|prior| equivalent(prior, child)) { return Err(format!("{path}: duplicate array item")); }
            }
        }
        let prefix = rules.get("prefixItems").and_then(Value::as_array);
        for (i, child) in array.iter().enumerate() {
            let rule = prefix.and_then(|p| p.get(i)).or_else(|| rules.get("items"));
            if let Some(rule) = rule { check_at(rule,child,&format!("{path}[{i}]"),depth+1)?; }
        }
    }
    Ok(())
}
