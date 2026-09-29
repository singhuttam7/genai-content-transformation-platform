import { getArtifact, getArtifacts } from "../../../services/api/artifacts";

function getResponseData(response) {
  return response?.data ?? response;
}

function getErrorMessage(error) {
  return (
    error?.response?.data?.error?.message ||
    error?.response?.data?.detail ||
    error?.message ||
    "Unable to load artifact data."
  );
}

function normalizeArtifactsResponse(response) {
  const data = getResponseData(response);

  if (Array.isArray(data)) {
    return {
      items: data,
      total: data.length,
    };
  }

  return {
    items: Array.isArray(data?.items) ? data.items : [],
    total:
      typeof data?.total === "number"
        ? data.total
        : Array.isArray(data?.items)
          ? data.items.length
          : 0,
  };
}

function normalizeArtifactResponse(response) {
  return getResponseData(response);
}

export async function fetchArtifacts() {
  try {
    const response = await getArtifacts();

    return normalizeArtifactsResponse(response);
  } catch (error) {
    const normalizedError = new Error(getErrorMessage(error));

    normalizedError.cause = error;

    throw normalizedError;
  }
}

export async function fetchArtifact(artifactId) {
  if (!artifactId) {
    throw new Error("Artifact ID is required.");
  }

  try {
    const response = await getArtifact(artifactId);

    return normalizeArtifactResponse(response);
  } catch (error) {
    const normalizedError = new Error(getErrorMessage(error));

    normalizedError.cause = error;

    throw normalizedError;
  }
}
