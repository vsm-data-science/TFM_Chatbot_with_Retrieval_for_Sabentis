import { useState } from "react";
import { MessagesContext } from "../context";

export const MessagesProvider = ({ children }) => {
  const state = useState([]);
  return (
    <MessagesContext.Provider value={state}>
      {children}
    </MessagesContext.Provider>
  );
};
