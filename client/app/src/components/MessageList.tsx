import { useContext } from "react";
import { MessagesContext } from "../context";
import { Message } from "./Message";

export function MessageList() {
  const [messages] = useContext(MessagesContext);
  return (
    <div className="flex flex-col gap-2 bg-slate-100 flex-1 mt-5 mx-5 rounded py-2 shadow-inner">
      {messages.map((message, index) => (
        <Message key={index} {...message} />
      ))}
    </div>
  );
}
