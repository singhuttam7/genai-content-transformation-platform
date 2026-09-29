export const transformationTypes = [
  {
    value: "advisory",
    label: "Advisory",
    description: "Create a clear, structured advisory for a target audience.",
  },
  {
    value: "executive_summary",
    label: "Executive Summary",
    description: "Condense source material into a decision-focused summary.",
  },
  {
    value: "social_media",
    label: "Social Media",
    description: "Turn source content into audience-ready social content.",
  },
  {
    value: "infographic",
    label: "Infographic",
    description: "Structure important information for visual communication.",
  },
  {
    value: "presentation",
    label: "Presentation",
    description:
      "Transform source material into a presentation-ready structure.",
  },
  {
    value: "video",
    label: "Video",
    description: "Prepare source material for a complete video package.",
  },
];

export const sourceTypes = [
  {
    value: "text",
    label: "Text",
    description: "Paste or write source content directly.",
    kind: "text",
  },
  {
    value: "prompt",
    label: "Prompt",
    description: "Provide an instruction or prompt as the source.",
    kind: "text",
  },
  {
    value: "url",
    label: "URL",
    description: "Fetch and process content from a web URL.",
    kind: "url",
  },
  {
    value: "pdf",
    label: "PDF",
    description: "Upload a PDF document.",
    kind: "file",
    accept: ".pdf,application/pdf",
  },
  {
    value: "docx",
    label: "DOCX",
    description: "Upload a Microsoft Word document.",
    kind: "file",
    accept:
      ".docx,application/vnd.openxmlformats-officedocument.wordprocessingml.document",
  },
  {
    value: "txt",
    label: "TXT",
    description: "Upload a plain text document.",
    kind: "file",
    accept: ".txt,text/plain",
  },
  {
    value: "markdown",
    label: "Markdown",
    description: "Upload a Markdown document.",
    kind: "file",
    accept: ".md,.markdown,text/markdown,text/plain",
  },
  {
    value: "image",
    label: "Image",
    description: "Upload an image containing visual information.",
    kind: "file",
    accept: "image/*",
  },
  {
    value: "audio",
    label: "Audio",
    description: "Upload an audio file for processing.",
    kind: "file",
    accept: "audio/*",
  },
  {
    value: "video",
    label: "Video",
    description: "Upload a video file for processing.",
    kind: "file",
    accept: "video/*",
  },
];

export const detailLevels = [
  {
    value: "concise",
    label: "Concise",
  },
  {
    value: "balanced",
    label: "Balanced",
  },
  {
    value: "detailed",
    label: "Detailed",
  },
];

export const commonLanguages = ["English", "Hindi"];

export const commonTones = [
  "Professional",
  "Formal",
  "Informative",
  "Conversational",
];

export const commonStyles = ["Clear", "Structured", "Technical", "Persuasive"];

export const commonAudiences = [
  "General public",
  "Executives",
  "Professionals",
  "Students",
  "Technical audience",
];
