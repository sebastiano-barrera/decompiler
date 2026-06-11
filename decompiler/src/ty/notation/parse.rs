use super::{Step, TypeBuilder, TypeID};

pub type ParseResult<T> = std::result::Result<T, ParseError>;
pub struct ParseError {
    offset: usize,
    description: String,
}
impl std::fmt::Debug for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "ParseError at offset {}: {}",
            self.offset, self.description
        )
    }
}

#[derive(Clone, Copy)]
struct Cursor<'a> {
    text: &'a str,
    offset: usize,
}

pub fn parse(text: &str) -> ParseResult<TypeBuilder> {
    let mut cursor = Cursor { text, offset: 0 };
    let mut program = Vec::new();
    parse_type(&mut cursor, &mut program)?;
    expect_end(&mut cursor)?;
    Ok(TypeBuilder { program })
}

fn expect_end(cursor: &mut Cursor) -> ParseResult<()> {
    // skip whitespace
    take_while(cursor, |c| c.is_whitespace());
    if !cursor.text.is_empty() {
        return Err(ParseError {
            offset: cursor.offset,
            description: format!("Expected end of input, got '{}'", cursor.text),
        });
    }

    Ok(())
}

fn parse_type(cursor: &mut Cursor, program: &mut Vec<Step>) -> ParseResult<()> {
    match next_token(cursor)? {
        Token::SquareOpen => {
            // array syntax: [N]T
            let count = match next_token(cursor)? {
                Token::LitUint(n) => n,
                other => return Err(error_unexpected_token(cursor, other)),
            };
            expect_token(cursor, Token::SquareClosed)?;
            parse_type(cursor, program)?;
            program.push(Step::Array { count });
        }
        Token::KwStruct => {
            expect_token(cursor, Token::BraceOpen)?;
            let mut count = 0;
            loop {
                parse_type(cursor, program)?;
                count += 1;
                match next_token(cursor)? {
                    Token::BraceClosed => break,
                    Token::Semicolon => {}
                    other => return Err(error_unexpected_token(cursor, other)),
                }
            }
            program.push(Step::Struct { count });
        }
        Token::KwFunc => {
            expect_token(cursor, Token::ParenOpen)?;
            let mut count = 0;
            loop {
                parse_type(cursor, program)?;
                count += 1;
                match next_token(cursor)? {
                    Token::ParenClosed => break,
                    Token::Comma => {}
                    other => return Err(error_unexpected_token(cursor, other)),
                }
            }
            parse_type(cursor, program)?;
            program.push(Step::Func { count });
        }
        Token::Star => {
            parse_type(cursor, program)?;
            program.push(Step::Ptr);
        }
        Token::KwInt8 => program.push(Step::Int { size_bytes: 1 }),
        Token::KwInt16 => program.push(Step::Int { size_bytes: 2 }),
        Token::KwInt32 => program.push(Step::Int { size_bytes: 4 }),
        Token::KwInt64 => program.push(Step::Int { size_bytes: 8 }),
        Token::KwUint8 => program.push(Step::Uint { size_bytes: 1 }),
        Token::KwUint16 => program.push(Step::Uint { size_bytes: 2 }),
        Token::KwUint32 => program.push(Step::Uint { size_bytes: 4 }),
        Token::KwUint64 => program.push(Step::Uint { size_bytes: 8 }),
        Token::KwFloat32 => program.push(Step::Float { size_bytes: 4 }),
        Token::KwFloat64 => program.push(Step::Float { size_bytes: 8 }),
        Token::Void => program.push(Step::Void),
        Token::Ident(tyid) => {
            program.push(Step::Ref { tyid });
        }
        other => return Err(error_unexpected_token(cursor, other)),
    }

    Ok(())
}

fn error_unexpected_token(cursor: &mut Cursor, other: Token) -> ParseError {
    ParseError {
        offset: cursor.offset,
        description: format!("Unexpected token: {:?}", other),
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Token {
    SquareOpen,
    SquareClosed,
    BraceOpen,
    BraceClosed,
    ParenOpen,
    ParenClosed,
    LitUint(u32),
    KwStruct,
    KwFunc,
    KwInt8,
    KwInt16,
    KwInt32,
    KwInt64,
    KwUint8,
    KwUint16,
    KwUint32,
    KwUint64,
    KwFloat32,
    KwFloat64,
    Ident(TypeID),
    Semicolon,
    Star,
    Comma,
    Void,
}

fn expect_token(cursor: &mut Cursor, expected: Token) -> ParseResult<()> {
    let token = next_token(cursor)?;
    if token == expected {
        Ok(())
    } else {
        Err(ParseError {
            offset: cursor.offset,
            description: format!("Expected token {:?}, got {:?}", expected, token),
        })
    }
}

fn next_token<'a>(cursor: &mut Cursor<'a>) -> ParseResult<Token> {
    // skip whitespace
    take_while(cursor, |c| c.is_whitespace());

    let cur_start = *cursor;

    match next_char(cursor) {
        None => {
            *cursor = cur_start;
            Err(ParseError {
                offset: cursor.offset,
                description: "Unexpected end of input".to_string(),
            })
        }
        Some('[') => Ok(Token::SquareOpen),
        Some(']') => Ok(Token::SquareClosed),
        Some('{') => Ok(Token::BraceOpen),
        Some('}') => Ok(Token::BraceClosed),
        Some('(') => Ok(Token::ParenOpen),
        Some(')') => Ok(Token::ParenClosed),
        Some('*') => Ok(Token::Star),
        Some(';') => Ok(Token::Semicolon),
        Some(',') => Ok(Token::Comma),
        Some('#') => {
            // e.g. #1234 -> Token::Ref(TypeID(1234))
            let start = cursor.offset;
            let num_str = take_while(cursor, |c| c.is_digit(10));
            let num = num_str.parse::<u64>().map_err(|_| ParseError {
                offset: start,
                description: format!("Invalid numeric type ID after '#': [{}]", num_str),
            })?;
            Ok(Token::Ident(TypeID(num)))
        }
        Some('0'..='9') => {
            *cursor = cur_start;
            let start = cursor.offset;
            let num_str = take_while(cursor, |c| c.is_digit(10));
            assert!(!num_str.is_empty());
            let num = num_str.parse::<u32>().map_err(|_| ParseError {
                offset: start,
                description: format!("Invalid number: [{}]", num_str),
            })?;
            Ok(Token::LitUint(num))
        }
        Some('a'..='z' | 'A'..='Z' | '_') => {
            *cursor = cur_start;
            let ident = take_while(cursor, |c| c.is_alphanumeric() || c == '_' || c == '\\');
            assert!(!ident.is_empty());
            match ident {
                "struct" => Ok(Token::KwStruct),
                "func" => Ok(Token::KwFunc),
                "i8" => Ok(Token::KwInt8),
                "i16" => Ok(Token::KwInt16),
                "i32" => Ok(Token::KwInt32),
                "i64" => Ok(Token::KwInt64),
                "u8" => Ok(Token::KwUint8),
                "u16" => Ok(Token::KwUint16),
                "u32" => Ok(Token::KwUint32),
                "u64" => Ok(Token::KwUint64),
                "f32" => Ok(Token::KwFloat32),
                "f64" => Ok(Token::KwFloat64),
                "void" => Ok(Token::Void),
                _ => {
                    *cursor = cur_start;
                    return Err(ParseError {
                        offset: cur_start.offset,
                        description: format!("Unrecognized keyword: '{}'", ident),
                    });
                }
            }
        }
        Some(other_ch) => {
            *cursor = cur_start;
            Err(ParseError {
                offset: cursor.offset,
                description: format!("Unexpected character: '{}'", other_ch),
            })
        }
    }
}

fn next_char(cursor: &mut Cursor) -> Option<char> {
    let ch = cursor.text.chars().next()?;
    cursor.offset += ch.len_utf8();
    cursor.text = &cursor.text[ch.len_utf8()..];
    Some(ch)
}

fn take_while<'a>(cursor: &mut Cursor<'a>, pred: impl Fn(char) -> bool) -> &'a str {
    let text = cursor.text;
    let mut count = 0;
    while let Some(c) = text[count..].chars().next() {
        if pred(c) {
            count += c.len_utf8();
        } else {
            break;
        }
    }

    cursor.text = &text[count..];
    &text[..count]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenizes_punctuation_and_keywords() {
        let cases = [
            ("[", Token::SquareOpen),
            ("]", Token::SquareClosed),
            ("{", Token::BraceOpen),
            ("}", Token::BraceClosed),
            (";", Token::Semicolon),
            ("*", Token::Star),
            ("struct", Token::KwStruct),
            ("i8", Token::KwInt8),
            ("i16", Token::KwInt16),
            ("i32", Token::KwInt32),
            ("i64", Token::KwInt64),
            ("u8", Token::KwUint8),
            ("u16", Token::KwUint16),
            ("u32", Token::KwUint32),
            ("u64", Token::KwUint64),
            ("f32", Token::KwFloat32),
            ("f64", Token::KwFloat64),
        ];

        for (input, expected) in cases {
            let mut cursor = Cursor {
                text: input,
                offset: 0,
            };
            assert_eq!(next_token(&mut cursor).unwrap(), expected, "input: {input}");
        }
    }

    #[test]
    fn tokenizes_numbers_and_identifiers() {
        let mut cursor = Cursor {
            text: "123 struct i32",
            offset: 0,
        };

        assert_eq!(next_token(&mut cursor).unwrap(), Token::LitUint(123));
        assert_eq!(next_token(&mut cursor).unwrap(), Token::KwStruct);
        assert_eq!(next_token(&mut cursor).unwrap(), Token::KwInt32);
    }
    #[test]
    fn tokenizes_numbers_and_identifiers_error() {
        let mut cursor = Cursor {
            text: "123 struct asd i32",
            offset: 0,
        };

        assert_eq!(next_token(&mut cursor).unwrap(), Token::LitUint(123));
        assert_eq!(next_token(&mut cursor).unwrap(), Token::KwStruct);
        assert!(matches!(
            next_token(&mut cursor).unwrap_err(),
            super::ParseError { .. }
        ));
    }

    #[test]
    fn tokenizer_skips_whitespace() {
        let mut cursor = Cursor {
            text: "\n\t  i8",
            offset: 0,
        };
        assert_eq!(next_token(&mut cursor).unwrap(), Token::KwInt8);
    }

    #[test]
    fn tokenizer_rejects_invalid_character() {
        let mut cursor = Cursor {
            text: "@",
            offset: 0,
        };
        let err = next_token(&mut cursor).unwrap_err();
        assert_eq!(err.offset, 0);
        assert!(err.description.contains("Unexpected character"));
    }

    #[test]
    fn parses_primitive_types() {
        let cases = [
            ("i8", vec![Step::Int { size_bytes: 1 }]),
            ("i16", vec![Step::Int { size_bytes: 2 }]),
            ("i32", vec![Step::Int { size_bytes: 4 }]),
            ("i64", vec![Step::Int { size_bytes: 8 }]),
            ("u8", vec![Step::Uint { size_bytes: 1 }]),
            ("u16", vec![Step::Uint { size_bytes: 2 }]),
            ("u32", vec![Step::Uint { size_bytes: 4 }]),
            ("u64", vec![Step::Uint { size_bytes: 8 }]),
            ("f32", vec![Step::Float { size_bytes: 4 }]),
            ("f64", vec![Step::Float { size_bytes: 8 }]),
        ];

        for (input, expected_program) in cases {
            let builder = parse(input).unwrap();
            assert_eq!(builder.program, expected_program, "input: {input}");
        }
    }

    #[test]
    fn parses_array_type() {
        let builder = parse("[3]u8").unwrap();
        assert_eq!(
            builder.program,
            vec![Step::Uint { size_bytes: 1 }, Step::Array { count: 3 }]
        );
    }

    #[test]
    fn parses_struct_type() {
        let builder = parse("struct { i8; u16; f32 }").unwrap();
        assert_eq!(
            builder.program,
            vec![
                Step::Int { size_bytes: 1 },
                Step::Uint { size_bytes: 2 },
                Step::Float { size_bytes: 4 },
                Step::Struct { count: 3 },
            ]
        );
    }

    #[test]
    fn parses_nested_composite_types() {
        let builder = parse("struct { [2]u8; struct { i16; u32 } }").unwrap();
        assert_eq!(
            builder.program,
            vec![
                Step::Uint { size_bytes: 1 },
                Step::Array { count: 2 },
                Step::Int { size_bytes: 2 },
                Step::Uint { size_bytes: 4 },
                Step::Struct { count: 2 },
                Step::Struct { count: 2 },
            ]
        );
    }

    #[test]
    fn parses_func() {
        parse("func(*#123459) void").unwrap();
    }

    #[test]
    fn rejects_invalid_input() {
        let err = parse("nope").unwrap_err();
        assert_eq!(err.offset, 0);
        assert!(err.description.contains("Unrecognized keyword"));
    }

    #[test]
    fn rejects_unterminated_struct() {
        let err = parse("struct { i8; u8").unwrap_err();
        assert!(err.description.contains("Unexpected end of input"));
    }
    #[test]
    fn rejects_trailing_tokens() {
        parse("struct { i8; u8 } xyz").unwrap_err();
    }
}
