use std::collections::HashMap;
use std::fs;
use std::io;
use haversine::reference_haversine;

#[derive(Debug, Clone, PartialEq)]
pub enum JsonValue {
    Object(HashMap<String, JsonValue>),
    Array(Vec<JsonValue>),
    Float(f64),
}

#[derive(Debug)]
pub enum ParseError {
    UnexpectedToken(String),
    UnexpectedEndOfInput,
    InvalidNumber(String),
    InvalidString(String),
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParseError::UnexpectedToken(msg) => write!(f, "Unexpected token: {}", msg),
            ParseError::UnexpectedEndOfInput => write!(f, "Unexpected end of input"),
            ParseError::InvalidNumber(msg) => write!(f, "Invalid number: {}", msg),
            ParseError::InvalidString(msg) => write!(f, "Invalid string: {}", msg),
        }
    }
}

impl std::error::Error for ParseError {}

/// Deserialize a JSON string into a JsonValue
/// Supports JSON objects, arrays, and float values
pub fn deserialize_json(input: &str) -> Result<JsonValue, ParseError> {
    let mut parser = Parser::new(input);
    parser.parse_value()
}

struct Parser<'a> {
    input: &'a str,
    pos: usize,
}

impl<'a> Parser<'a> {
    fn new(input: &'a str) -> Self {
        Parser { input, pos: 0 }
    }

    fn skip_whitespace(&mut self) {
        while self.pos < self.input.len() {
            match self.input.as_bytes()[self.pos] {
                b' ' | b'\t' | b'\n' | b'\r' => self.pos += 1,
                _ => break,
            }
        }
    }

    fn peek_char(&self) -> Option<u8> {
        if self.pos < self.input.len() {
            Some(self.input.as_bytes()[self.pos])
        } else {
            None
        }
    }

    fn consume_char(&mut self) -> Option<u8> {
        if self.pos < self.input.len() {
            let ch = self.input.as_bytes()[self.pos];
            self.pos += 1;
            Some(ch)
        } else {
            None
        }
    }

    fn expect_char(&mut self, expected: u8) -> Result<(), ParseError> {
        self.skip_whitespace();
        match self.consume_char() {
            Some(ch) if ch == expected => Ok(()),
            Some(ch) => Err(ParseError::UnexpectedToken(format!(
                "Expected '{}', got '{}'",
                expected as char, ch as char
            ))),
            None => Err(ParseError::UnexpectedEndOfInput),
        }
    }

    fn parse_value(&mut self) -> Result<JsonValue, ParseError> {
        self.skip_whitespace();
        match self.peek_char() {
            Some(b'{') => self.parse_object(),
            Some(b'[') => self.parse_array(),
            Some(b'-') | Some(b'0'..=b'9') => self.parse_number(),
            Some(ch) => Err(ParseError::UnexpectedToken(format!(
                "Unexpected character: '{}'",
                ch as char
            ))),
            None => Err(ParseError::UnexpectedEndOfInput),
        }
    }

    fn parse_object(&mut self) -> Result<JsonValue, ParseError> {
        self.expect_char(b'{')?;
        let mut map = HashMap::new();

        self.skip_whitespace();
        if self.peek_char() == Some(b'}') {
            self.consume_char();
            return Ok(JsonValue::Object(map));
        }

        loop {
            self.skip_whitespace();
            let key = self.parse_string()?;

            self.expect_char(b':')?;

            let value = self.parse_value()?;
            map.insert(key, value);

            self.skip_whitespace();
            match self.peek_char() {
                Some(b',') => {
                    self.consume_char();
                }
                Some(b'}') => {
                    self.consume_char();
                    break;
                }
                Some(ch) => {
                    return Err(ParseError::UnexpectedToken(format!(
                        "Expected ',' or '}}', got '{}'",
                        ch as char
                    )))
                }
                None => return Err(ParseError::UnexpectedEndOfInput),
            }
        }

        Ok(JsonValue::Object(map))
    }

    fn parse_array(&mut self) -> Result<JsonValue, ParseError> {
        self.expect_char(b'[')?;
        let mut elements = Vec::new();

        self.skip_whitespace();
        if self.peek_char() == Some(b']') {
            self.consume_char();
            return Ok(JsonValue::Array(elements));
        }

        loop {
            let value = self.parse_value()?;
            elements.push(value);

            self.skip_whitespace();
            match self.peek_char() {
                Some(b',') => {
                    self.consume_char();
                }
                Some(b']') => {
                    self.consume_char();
                    break;
                }
                Some(ch) => {
                    return Err(ParseError::UnexpectedToken(format!(
                        "Expected ',' or ']', got '{}'",
                        ch as char
                    )))
                }
                None => return Err(ParseError::UnexpectedEndOfInput),
            }
        }

        Ok(JsonValue::Array(elements))
    }

    fn parse_string(&mut self) -> Result<String, ParseError> {
        self.skip_whitespace();
        self.expect_char(b'"')?;

        let start = self.pos;
        while self.pos < self.input.len() {
            let ch = self.input.as_bytes()[self.pos];
            if ch == b'"' {
                let result = self.input[start..self.pos].to_string();
                self.pos += 1;
                return Ok(result);
            }
            if ch == b'\\' {
                return Err(ParseError::InvalidString(
                    "Escape sequences not supported".to_string(),
                ));
            }
            self.pos += 1;
        }

        Err(ParseError::UnexpectedEndOfInput)
    }

    fn parse_number(&mut self) -> Result<JsonValue, ParseError> {
        self.skip_whitespace();
        let start = self.pos;

        // Optional negative sign
        if self.peek_char() == Some(b'-') {
            self.consume_char();
        }

        // Integer part
        if !self.consume_digits() {
            return Err(ParseError::InvalidNumber("Expected digit".to_string()));
        }

        // Optional decimal part
        if self.peek_char() == Some(b'.') {
            self.consume_char();
            if !self.consume_digits() {
                return Err(ParseError::InvalidNumber(
                    "Expected digit after decimal point".to_string(),
                ));
            }
        }

        // Optional exponent
        if let Some(b'e') | Some(b'E') = self.peek_char() {
            self.consume_char();
            if let Some(b'+') | Some(b'-') = self.peek_char() {
                self.consume_char();
            }
            if !self.consume_digits() {
                return Err(ParseError::InvalidNumber(
                    "Expected digit in exponent".to_string(),
                ));
            }
        }

        let num_str = &self.input[start..self.pos];
        match num_str.parse::<f64>() {
            Ok(num) => Ok(JsonValue::Float(num)),
            Err(_) => Err(ParseError::InvalidNumber(format!("Invalid number: {}", num_str))),
        }
    }

    fn consume_digits(&mut self) -> bool {
        let start = self.pos;
        while self.pos < self.input.len() {
            match self.input.as_bytes()[self.pos] {
                b'0'..=b'9' => self.pos += 1,
                _ => break,
            }
        }
        self.pos > start
    }
}

/// Extract point pairs from parsed JSON and compute the sum of Haversine distances
fn compute_haversine_sum_from_json(json: &JsonValue) -> Result<f64, String> {
    let earth_radius = 6372.8;
    
    // Extract the "pairs" array from the root object
    let pairs_array = match json {
        JsonValue::Object(map) => {
            match map.get("pairs") {
                Some(JsonValue::Array(arr)) => arr,
                Some(_) => return Err("'pairs' field is not an array".to_string()),
                None => return Err("Missing 'pairs' field in JSON".to_string()),
            }
        }
        _ => return Err("Root JSON value is not an object".to_string()),
    };
    
    let mut sum = 0.0;
    
    // Process each pair in the array
    for (i, pair) in pairs_array.iter().enumerate() {
        let pair_obj = match pair {
            JsonValue::Object(map) => map,
            _ => return Err(format!("Pair at index {} is not an object", i)),
        };
        
        // Extract x0, y0, x1, y1 from the pair object
        let x0 = extract_float(pair_obj, "x0", i)?;
        let y0 = extract_float(pair_obj, "y0", i)?;
        let x1 = extract_float(pair_obj, "x1", i)?;
        let y1 = extract_float(pair_obj, "y1", i)?;
        
        let distance = reference_haversine([x0, y0], [x1, y1], earth_radius);
        sum += distance;
    }
    
    Ok(sum)
}

/// Helper function to extract a float value from a HashMap
fn extract_float(map: &HashMap<String, JsonValue>, key: &str, pair_index: usize) -> Result<f64, String> {
    match map.get(key) {
        Some(JsonValue::Float(value)) => Ok(*value),
        Some(_) => Err(format!("Field '{}' in pair {} is not a float", key, pair_index)),
        None => Err(format!("Missing field '{}' in pair {}", key, pair_index)),
    }
}

fn main() {
    println!("Haversine Distance Sum Calculator");
    
    // Read filename from command line arguments or stdin
    let filename = std::env::args().nth(1).unwrap_or_else(|| {
        println!("Enter JSON filename:");
        let mut input = String::new();
        io::stdin()
            .read_line(&mut input)
            .expect("Failed to read input");
        input.trim().to_string()
    });
    
    // Read the JSON file
    let json_content = match fs::read_to_string(&filename) {
        Ok(content) => content,
        Err(e) => {
            eprintln!("Error reading file '{}': {}", filename, e);
            return;
        }
    };
    
    // Parse the JSON
    let parsed = match deserialize_json(&json_content) {
        Ok(value) => value,
        Err(e) => {
            eprintln!("Error parsing JSON: {}", e);
            return;
        }
    };
    
    // Compute the sum of Haversine distances
    match compute_haversine_sum_from_json(&parsed) {
        Ok(sum) => {
            println!("Sum of Haversine distances: {}", sum);
        }
        Err(e) => {
            eprintln!("Error computing Haversine sum: {}", e);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_float() {
        let result = deserialize_json("42.5").unwrap();
        assert_eq!(result, JsonValue::Float(42.5));
    }

    #[test]
    fn test_parse_negative_float() {
        let result = deserialize_json("-3.14").unwrap();
        assert_eq!(result, JsonValue::Float(-3.14));
    }

    #[test]
    fn test_parse_empty_object() {
        let result = deserialize_json("{}").unwrap();
        assert_eq!(result, JsonValue::Object(HashMap::new()));
    }

    #[test]
    fn test_parse_simple_object() {
        let result = deserialize_json(r#"{"x": 1.5, "y": 2.5}"#).unwrap();
        
        if let JsonValue::Object(map) = result {
            assert_eq!(map.get("x"), Some(&JsonValue::Float(1.5)));
            assert_eq!(map.get("y"), Some(&JsonValue::Float(2.5)));
        } else {
            panic!("Expected Object");
        }
    }

    #[test]
    fn test_parse_nested_object() {
        let result = deserialize_json(r#"{"outer": {"inner": 42.0}}"#).unwrap();
        
        if let JsonValue::Object(map) = result {
            if let Some(JsonValue::Object(inner_map)) = map.get("outer") {
                assert_eq!(inner_map.get("inner"), Some(&JsonValue::Float(42.0)));
            } else {
                panic!("Expected nested object");
            }
        } else {
            panic!("Expected Object");
        }
    }

    #[test]
    fn test_parse_with_whitespace() {
        let result = deserialize_json(r#"  {  "x"  :  1.5  }  "#).unwrap();
        
        if let JsonValue::Object(map) = result {
            assert_eq!(map.get("x"), Some(&JsonValue::Float(1.5)));
        } else {
            panic!("Expected Object");
        }
    }

    #[test]
    fn test_parse_scientific_notation() {
        let result = deserialize_json("1.5e10").unwrap();
        assert_eq!(result, JsonValue::Float(1.5e10));
    }

    #[test]
    fn test_parse_error_invalid_json() {
        let result = deserialize_json("{invalid}");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_empty_array() {
        let result = deserialize_json("[]").unwrap();
        assert_eq!(result, JsonValue::Array(vec![]));
    }

    #[test]
    fn test_parse_simple_array() {
        let result = deserialize_json("[1.0, 2.5, 3.14]").unwrap();
        
        if let JsonValue::Array(elements) = result {
            assert_eq!(elements.len(), 3);
            assert_eq!(elements[0], JsonValue::Float(1.0));
            assert_eq!(elements[1], JsonValue::Float(2.5));
            assert_eq!(elements[2], JsonValue::Float(3.14));
        } else {
            panic!("Expected Array");
        }
    }

    #[test]
    fn test_parse_nested_array() {
        let result = deserialize_json("[[1.0, 2.0], [3.0, 4.0]]").unwrap();
        
        if let JsonValue::Array(outer) = result {
            assert_eq!(outer.len(), 2);
            
            if let JsonValue::Array(inner1) = &outer[0] {
                assert_eq!(inner1[0], JsonValue::Float(1.0));
                assert_eq!(inner1[1], JsonValue::Float(2.0));
            } else {
                panic!("Expected nested array");
            }
            
            if let JsonValue::Array(inner2) = &outer[1] {
                assert_eq!(inner2[0], JsonValue::Float(3.0));
                assert_eq!(inner2[1], JsonValue::Float(4.0));
            } else {
                panic!("Expected nested array");
            }
        } else {
            panic!("Expected Array");
        }
    }

    #[test]
    fn test_parse_object_with_array() {
        let result = deserialize_json(r#"{"values": [1.0, 2.0, 3.0]}"#).unwrap();
        
        if let JsonValue::Object(map) = result {
            if let Some(JsonValue::Array(arr)) = map.get("values") {
                assert_eq!(arr.len(), 3);
                assert_eq!(arr[0], JsonValue::Float(1.0));
                assert_eq!(arr[1], JsonValue::Float(2.0));
                assert_eq!(arr[2], JsonValue::Float(3.0));
            } else {
                panic!("Expected array value");
            }
        } else {
            panic!("Expected Object");
        }
    }

    #[test]
    fn test_parse_array_with_objects() {
        let result = deserialize_json(r#"[{"x": 1.0}, {"y": 2.0}]"#).unwrap();
        
        if let JsonValue::Array(arr) = result {
            assert_eq!(arr.len(), 2);
            
            if let JsonValue::Object(obj1) = &arr[0] {
                assert_eq!(obj1.get("x"), Some(&JsonValue::Float(1.0)));
            } else {
                panic!("Expected object in array");
            }
            
            if let JsonValue::Object(obj2) = &arr[1] {
                assert_eq!(obj2.get("y"), Some(&JsonValue::Float(2.0)));
            } else {
                panic!("Expected object in array");
            }
        } else {
            panic!("Expected Array");
        }
    }

    #[test]
    fn test_parse_array_with_whitespace() {
        let result = deserialize_json("  [  1.0  ,  2.0  ]  ").unwrap();
        
        if let JsonValue::Array(arr) = result {
            assert_eq!(arr.len(), 2);
            assert_eq!(arr[0], JsonValue::Float(1.0));
            assert_eq!(arr[1], JsonValue::Float(2.0));
        } else {
            panic!("Expected Array");
        }
    }
}
