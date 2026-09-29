import { getExecution, getExecutions } from "../../../services/api/executions";

function getResponseData(response) {
  return response?.data ?? response;
}

function getErrorMessage(error) {
  return (
    error?.response?.data?.error?.message ||
    error?.response?.data?.detail ||
    error?.message ||
    "Unable to load execution data."
  );
}

function normalizeExecutionsResponse(response) {
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

function normalizeExecutionResponse(response) {
  return getResponseData(response);
}

export async function fetchExecutions() {
  try {
    const response = await getExecutions();

    return normalizeExecutionsResponse(response);
  } catch (error) {
    const normalizedError = new Error(getErrorMessage(error));

    normalizedError.cause = error;

    throw normalizedError;
  }
}

export async function fetchExecution(executionId) {
  if (!executionId) {
    throw new Error("Execution ID is required.");
  }

  try {
    const response = await getExecution(executionId);

    return normalizeExecutionResponse(response);
  } catch (error) {
    const normalizedError = new Error(getErrorMessage(error));

    normalizedError.cause = error;

    throw normalizedError;
  }
}
