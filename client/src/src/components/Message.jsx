import { tv } from "tailwind-variants";
import { AUTHOR_TYPES } from "../types";

const MessageStyleClasses = tv({
  slots: {
    container: "flex gap-2 p-2 bg-white rounded w-fit mx-2 shadow",
    text: "",
  },
  variants: {
    type: {
      [AUTHOR_TYPES.USER]: {
        container: "self-end bg-blue-500",
        text: "text-white",
      },
      [AUTHOR_TYPES.BOT]: {
        text: "text-black",
      },
    },
  },
});

export function Message({ author, body }) {
  const { container, text } = MessageStyleClasses({ type: author });
  return (
    <div className={container()}>
      <div className={text()}>{body}</div>
    </div>
  );
}
