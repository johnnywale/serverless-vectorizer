// PDF text extraction utilities

use std::path::Path;
use thiserror::Error;

/// Errors that can occur during PDF processing
#[derive(Error, Debug)]
pub enum PdfError {
    #[error("Failed to read PDF file: {0}")]
    ReadError(String),

    #[error("Failed to extract text from PDF: {0}")]
    ExtractionError(String),

    #[error("PDF file is empty or contains no extractable text")]
    EmptyDocument,

    #[error("Invalid PDF format: {0}")]
    InvalidFormat(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

/// Result of PDF text extraction
#[derive(Debug, Clone)]
pub struct PdfDocument {
    /// Extracted text content
    pub text: String,
    /// Number of pages in the document
    pub page_count: usize,
    /// Text extracted per page
    pub pages: Vec<String>,
}

impl PdfDocument {
    /// Get all text as a single string
    pub fn full_text(&self) -> &str {
        &self.text
    }

    /// Get text from a specific page (0-indexed)
    pub fn page_text(&self, page: usize) -> Option<&str> {
        self.pages.get(page).map(|s| s.as_str())
    }

    /// Check if document is empty
    pub fn is_empty(&self) -> bool {
        self.text.trim().is_empty()
    }

    /// Get word count estimate
    pub fn word_count(&self) -> usize {
        self.text.split_whitespace().count()
    }
}

/// Extract text from a PDF file
pub fn extract_text_from_file<P: AsRef<Path>>(path: P) -> Result<PdfDocument, PdfError> {
    let path = path.as_ref();

    // Read PDF bytes
    let bytes = std::fs::read(path)
        .map_err(|e| PdfError::ReadError(format!("{}: {}", path.display(), e)))?;

    extract_text_from_bytes(&bytes)
}

/// Extract text from PDF bytes
pub fn extract_text_from_bytes(bytes: &[u8]) -> Result<PdfDocument, PdfError> {
    // Use pdf-extract for text extraction
    let text = pdf_extract::extract_text_from_mem(bytes)
        .map_err(|e| PdfError::ExtractionError(e.to_string()))?;

    // Try to get page count using lopdf
    let page_count = match lopdf::Document::load_mem(bytes) {
        Ok(doc) => doc.get_pages().len(),
        Err(_) => 1, // Default to 1 if we can't read page count
    };

    // Clean up the text
    let text = clean_extracted_text(&text);

    if text.trim().is_empty() {
        return Err(PdfError::EmptyDocument);
    }

    // For now, we don't have per-page extraction with pdf-extract
    // so we put all text in a single "page"
    let pages = vec![text.clone()];

    Ok(PdfDocument {
        text,
        page_count,
        pages,
    })
}

/// Extract text from PDF with page-by-page extraction
pub fn extract_text_by_pages<P: AsRef<Path>>(path: P) -> Result<PdfDocument, PdfError> {
    let path = path.as_ref();
    let bytes = std::fs::read(path)
        .map_err(|e| PdfError::ReadError(format!("{}: {}", path.display(), e)))?;

    extract_text_by_pages_from_bytes(&bytes)
}

/// Extract text from PDF bytes with page-by-page extraction
pub fn extract_text_by_pages_from_bytes(bytes: &[u8]) -> Result<PdfDocument, PdfError> {
    let doc = lopdf::Document::load_mem(bytes)
        .map_err(|e| PdfError::InvalidFormat(e.to_string()))?;

    let pages_map = doc.get_pages();
    let page_count = pages_map.len();

    let mut pages = Vec::with_capacity(page_count);
    let mut all_text = String::new();

    // Extract text from each page
    for page_num in 1..=page_count as u32 {
        let page_text = match doc.extract_text(&[page_num]) {
            Ok(text) => clean_extracted_text(&text),
            Err(_) => String::new(),
        };

        if !page_text.is_empty() {
            if !all_text.is_empty() {
                all_text.push_str("\n\n");
            }
            all_text.push_str(&page_text);
        }
        pages.push(page_text);
    }

    if all_text.trim().is_empty() {
        return Err(PdfError::EmptyDocument);
    }

    Ok(PdfDocument {
        text: all_text,
        page_count,
        pages,
    })
}

/// Clean up extracted text
fn clean_extracted_text(text: &str) -> String {
    text.lines()
        .map(|line| line.trim())
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
        .replace("\x00", "") // Remove null bytes
        .replace('\u{FFFD}', "") // Remove replacement characters
}

/// Check if a file is a PDF based on extension
pub fn is_pdf_file<P: AsRef<Path>>(path: P) -> bool {
    path.as_ref()
        .extension()
        .map(|ext| ext.to_ascii_lowercase() == "pdf")
        .unwrap_or(false)
}

/// Check if bytes appear to be a PDF (magic number check)
pub fn is_pdf_bytes(bytes: &[u8]) -> bool {
    bytes.starts_with(b"%PDF")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_pdf_file() {
        assert!(is_pdf_file("document.pdf"));
        assert!(is_pdf_file("document.PDF"));
        assert!(!is_pdf_file("document.txt"));
        assert!(!is_pdf_file("document"));
    }

    #[test]
    fn test_is_pdf_bytes() {
        assert!(is_pdf_bytes(b"%PDF-1.4"));
        assert!(is_pdf_bytes(b"%PDF-2.0"));
        assert!(!is_pdf_bytes(b"Hello World"));
        assert!(!is_pdf_bytes(b""));
    }

    #[test]
    fn test_clean_extracted_text() {
        let dirty = "  Hello  \n\n  World  \n  ";
        let clean = clean_extracted_text(dirty);
        assert_eq!(clean, "Hello\nWorld");
    }
}
