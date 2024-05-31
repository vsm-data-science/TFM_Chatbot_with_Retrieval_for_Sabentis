import { Dispatch, SetStateAction, createContext } from "react";
import { Message } from "./types";

export const MessagesContext = createContext<
  [Message[], Dispatch<SetStateAction<Message[]>>]
>([[], () => {}]);
