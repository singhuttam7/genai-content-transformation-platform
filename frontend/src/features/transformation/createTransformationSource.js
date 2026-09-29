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
  switch (sourceType) {
    case "text":
      return createTextSource({
        content,
        title,
        projectId,
      });

    case "prompt":
      return createPromptSource({
        content,
        title,
        projectId,
      });

    case "url":
      return createUrlSource({
        url,
        title,
        projectId,
      });

    case "pdf":
    case "docx":
    case "txt":
    case "markdown":
    case "image":
    case "audio":
    case "video":
      if (!file) {
        throw new Error(`Please select a ${sourceType.toUpperCase()} file.`);
      }

      return createFileSource({
        file,
        inputType: sourceType,
        title,
        projectId,
      });

    default:
      throw new Error(
        `Source type "${sourceType}" is not currently supported by the source API.`,
      );
  }
}
