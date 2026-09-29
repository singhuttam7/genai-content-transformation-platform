import { apiClient } from "./client";

/**
 * Create an artifact.
 *
 * @param {Object} artifact
 * @param {string} artifact.transformationId
 * @param {string} artifact.executionId
 * @param {string} artifact.artifactType
 * @param {string} [artifact.title]
 * @param {*} [artifact.content]
 * @param {string} [artifact.storageUri]
 * @param {Object} [artifact.metadata]
 * @param {string} [artifact.status]
 * @returns {Promise<Object>}
 */
export function createArtifact({
  transformationId,
  executionId,
  artifactType,
  title,
  content,
  storageUri,
  metadata = {},
  status = "GENERATED",
}) {
  return apiClient.post("/artifacts", {
    transformation_id: transformationId,
    execution_id: executionId,
    artifact_type: artifactType,
    title,
    content,
    storage_uri: storageUri,
    metadata,
    status,
  });
}

/**
 * Get an artifact by ID.
 *
 * @param {string} artifactId
 * @returns {Promise<Object>}
 */
export function getArtifact(artifactId) {
  return apiClient.get(`/artifacts/${artifactId}`);
}

/**
 * List artifacts.
 *
 * @param {Object} [params]
 * @returns {Promise<Object|Array>}
 */
export function getArtifacts(params = {}) {
  const searchParams = new URLSearchParams();

  Object.entries(params).forEach(([key, value]) => {
    if (value !== undefined && value !== null && value !== "") {
      searchParams.set(key, String(value));
    }
  });

  const query = searchParams.toString();

  return apiClient.get(query ? `/artifacts?${query}` : "/artifacts");
}
