import { FC, PropsWithChildren, useState } from "react";
import { MessagesContext } from "../context";
import { AUTHOR_TYPES, Message } from "../types";

export const MessagesProvider: FC<PropsWithChildren> = ({ children }) => {
  const state = useState<Message[]>([
    {
      author: AUTHOR_TYPES.BOT,
      body: "¡Hola! Soy un chatbot de Sabentis y estoy aquí para ayudarte",
    },
  ]);
  return (
    <MessagesContext.Provider value={state}>
      {children}
    </MessagesContext.Provider>
  );
};
