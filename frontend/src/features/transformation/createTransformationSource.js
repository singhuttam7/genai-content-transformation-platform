import {
  createFileSource,
  createPromptSource,
  createTextSource,
  createUrlSource,
} from "../../services/api/sources";

export function createTransformationSource({
  sourceType,
  title,
  content,
  url,
  file,
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

  if (sourceType === "file") {
    if (!file) {
      throw new Error("Please select a PDF file.");
    }

    return createFileSource({
      file,
      title,
      projectId,
    });
  }

  throw new Error(
    `Source type "${sourceType}" is not currently supported by the source API.`,
  );
}
