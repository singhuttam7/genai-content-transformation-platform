const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000/api/v1";

/**
 * Central HTTP client for the GenAI Content Transformation Platform.
 *
 * All frontend API services should use this client instead of
 * calling fetch() directly.
 */

async function request(path, options = {}) {
  const { method = "GET", body, headers = {}, ...fetchOptions } = options;

  const requestHeaders = {
    Accept: "application/json",
    ...headers,
  };

  const requestOptions = {
    method,
    headers: requestHeaders,
    ...fetchOptions,
  };

  if (body !== undefined) {
    const isFormData =
      typeof FormData !== "undefined" && body instanceof FormData;

    if (isFormData) {
      /*
       * Do not set Content-Type manually for FormData.
       *
       * The browser automatically creates:
       *
       * multipart/form-data; boundary=...
       *
       * FastAPI needs this boundary to correctly parse
       * UploadFile and Form fields.
       */
      delete requestHeaders["Content-Type"];
      delete requestHeaders["content-type"];

      requestOptions.body = body;
    } else {
      requestHeaders["Content-Type"] = "application/json";
      requestOptions.body = JSON.stringify(body);
    }
  }

  const response = await fetch(`${API_BASE_URL}${path}`, requestOptions);

  let responseData = null;

  const contentType = response.headers.get("content-type") || "";

  if (contentType.includes("application/json")) {
    responseData = await response.json();
  } else {
    responseData = await response.text();
  }

  if (!response.ok) {
    const message =
      typeof responseData === "object" &&
      responseData !== null &&
      "detail" in responseData
        ? responseData.detail
        : `API request failed with status ${response.status}.`;

    throw new Error(
      typeof message === "string" ? message : JSON.stringify(message),
    );
  }

  return responseData;
}

export const apiClient = {
  get(path, options = {}) {
    return request(path, {
      ...options,
      method: "GET",
    });
  },

  post(path, body, options = {}) {
    return request(path, {
      ...options,
      method: "POST",
      body,
    });
  },

  put(path, body, options = {}) {
    return request(path, {
      ...options,
      method: "PUT",
      body,
    });
  },

  patch(path, body, options = {}) {
    return request(path, {
      ...options,
      method: "PATCH",
      body,
    });
  },

  delete(path, options = {}) {
    return request(path, {
      ...options,
      method: "DELETE",
    });
  },
};

export { API_BASE_URL };
