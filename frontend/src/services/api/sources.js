import { apiClient } from "./client";

/**
 * Create and ingest a source through the backend.
 *
 * Supported input types are defined by the backend:
 * text, prompt, url, pdf, docx, txt, markdown,
 * image, audio, and video.
 *
 * @param {Object} source
 * @returns {Promise<Object>}
 */
export function createSource(source) {
  return apiClient.post("/sources", source);
}

/**
 * Create a text source.
 *
 * @param {Object} params
 * @param {string} params.content
 * @param {string} [params.title]
 * @param {string} [params.projectId]
 * @param {Object} [params.metadata]
 */
export function createTextSource({ content, title, projectId, metadata = {} }) {
  return createSource({
    input_type: "text",
    content,
    title,
    project_id: projectId,
    metadata,
  });
}

/**
 * Create a prompt source.
 *
 * @param {Object} params
 * @param {string} params.content
 * @param {string} [params.title]
 * @param {string} [params.projectId]
 * @param {Object} [params.metadata]
 */
export function createPromptSource({
  content,
  title,
  projectId,
  metadata = {},
}) {
  return createSource({
    input_type: "prompt",
    content,
    title,
    project_id: projectId,
    metadata,
  });
}

/**
 * Create a URL source.
 *
 * @param {Object} params
 * @param {string} params.url
 * @param {string} [params.title]
 * @param {string} [params.projectId]
 * @param {Object} [params.metadata]
 */
export function createUrlSource({ url, title, projectId, metadata = {} }) {
  return createSource({
    input_type: "url",
    url,
    title,
    project_id: projectId,
    metadata,
  });
}

/**
 * Upload and ingest a PDF source.
 *
 * Uses multipart/form-data because the backend expects
 * the actual PDF file as an UploadFile.
 *
 * @param {Object} params
 * @param {File} params.file
 * @param {string} [params.title]
 * @param {string} params.projectId
 * @param {Object} [params.metadata]
 */
export function createFileSource({ file, title, projectId, metadata = {} }) {
  if (!file) {
    throw new Error("Please select a PDF file.");
  }

  const formData = new FormData();

  formData.append("file", file);
  formData.append("project_id", projectId);

  if (title) {
    formData.append("title", title);
  }

  formData.append("metadata", JSON.stringify(metadata));

  return apiClient.post("/sources/upload", formData);
}
