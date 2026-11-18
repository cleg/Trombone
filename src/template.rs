use std::fmt;

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum RenderError {
    UnmatchedBrace,
    UnknownPlaceholder(String),
    MissingPlaceholder(String),
}

impl fmt::Display for RenderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnmatchedBrace => f.write_str("unmatched brace"),
            Self::UnknownPlaceholder(name) => write!(f, "unknown placeholder {{{name}}}"),
            Self::MissingPlaceholder(name) => write!(f, "missing required placeholder {{{name}}}"),
        }
    }
}

impl std::error::Error for RenderError {}

/// Substitute named placeholders in one pass. Doubled braces produce literal braces.
/// Every name in `required` must appear as a placeholder in the template.
pub(crate) fn render(
    template: &str,
    values: &[(&str, &str)],
    required: &[&str],
) -> Result<String, RenderError> {
    let mut input = template;
    let mut output = String::with_capacity(template.len());
    let mut missing = required.to_vec();

    while let Some(offset) = input.find(['{', '}']) {
        output.push_str(&input[..offset]);
        input = &input[offset..];
        if input.starts_with("{{") || input.starts_with("}}") {
            output.push_str(&input[..1]);
            input = &input[2..];
            continue;
        }

        let Some((name, rest)) = input.strip_prefix('{').and_then(|s| s.split_once('}')) else {
            return Err(RenderError::UnmatchedBrace);
        };
        let Some((_, value)) = values.iter().find(|(key, _)| *key == name) else {
            return Err(RenderError::UnknownPlaceholder(name.to_owned()));
        };
        output.push_str(value);
        missing.retain(|key| *key != name);
        input = rest;
    }

    if let Some(name) = missing.first() {
        return Err(RenderError::MissingPlaceholder((*name).to_owned()));
    }
    output.push_str(input);
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::{RenderError, render};

    #[test]
    fn substitutes_repeated_names_without_reinterpreting_values() {
        let values = [
            ("recipient", "Оля {item}"),
            ("item", "книга"),
            ("day", "понеділок"),
        ];
        let template = "{recipient}: {item}, {day}. Ще раз: {item}.";
        assert_eq!(
            render(template, &values, &["recipient", "item", "day"]).unwrap(),
            "Оля {item}: книга, понеділок. Ще раз: книга."
        );
    }

    #[test]
    fn handles_literal_braces_and_optional_placeholders() {
        assert_eq!(
            render("{{{name}}} {{literal}}", &[("name", "x")], &[]).unwrap(),
            "{x} {literal}"
        );
        assert_eq!(
            render("plain text", &[("unused", "x")], &[]).unwrap(),
            "plain text"
        );
    }

    #[test]
    fn rejects_unknown_and_missing_placeholders() {
        assert_eq!(
            render("{other}", &[], &[]),
            Err(RenderError::UnknownPlaceholder("other".into()))
        );
        assert_eq!(
            render(
                "{first} {{second}}",
                &[("first", "x"), ("second", "y")],
                &["first", "second"]
            ),
            Err(RenderError::MissingPlaceholder("second".into()))
        );
    }

    #[test]
    fn rejects_unmatched_braces() {
        for template in ["prefix {name", "prefix }", "{name}}"] {
            assert_eq!(
                render(template, &[("name", "x")], &[]),
                Err(RenderError::UnmatchedBrace)
            );
        }
    }
}
