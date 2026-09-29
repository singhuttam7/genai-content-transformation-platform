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
