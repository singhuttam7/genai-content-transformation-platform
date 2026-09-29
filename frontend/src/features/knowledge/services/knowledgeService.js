import { getSources } from "../../../services/api/sources";

function getResponseData(response) {
  return response?.data ?? response;
}

function getErrorMessage(error) {
  return (
    error?.response?.data?.error?.message ||
    error?.response?.data?.detail ||
    error?.message ||
    "Unable to load knowledge data."
  );
}

function normalizeSourcesResponse(response) {
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

export async function fetchKnowledgeSources() {
  try {
    const response = await getSources();

    return normalizeSourcesResponse(response);
  } catch (error) {
    const normalizedError = new Error(getErrorMessage(error));

    normalizedError.cause = error;

    throw normalizedError;
  }
}
