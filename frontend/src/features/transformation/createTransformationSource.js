import {
  createPromptSource,
  createTextSource,
  createUrlSource,
} from "../../services/api/sources";

export function createTransformationSource({
  sourceType,
  title,
  content,
  url,
  projectId,
}) {
  if (sourceType === "text") {
    return createTextSource({
      content,
      title,
      projectId,
    });
  }

  if (sourceType === "prompt") {
    return createPromptSource({
      content,
      title,
      projectId,
    });
  }

  if (sourceType === "url") {
    return createUrlSource({
      url,
      title,
      projectId,
    });
  }

  throw new Error(
    `Source type "${sourceType}" is not currently supported by the direct source API.`,
  );
}
