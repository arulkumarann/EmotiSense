/** Short display names for the timeline model kinds a session can record. */
export const MODEL_LABEL: Record<string, string> = {
  bilstm: "BiLSTM",
  transformer: "Transformer",
  mlp: "No-context baseline",
  lstm: "LSTM (one-way)",
  transformer_causal: "Transformer (one-way)",
};

export const modelLabel = (kind: string) => MODEL_LABEL[kind] ?? kind;
