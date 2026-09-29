import { apiClient } from "./client";

export function createTransformation(transformation) {
  return apiClient.post("/transformations", transformation);
}
