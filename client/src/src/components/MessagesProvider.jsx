import { useState } from "react";
import { MessagesContext } from "../context";
import { AUTHOR_TYPES } from "../types";

export const MessagesProvider = ({ children }) => {
  const state = useState([
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
