import { useForm } from "react-hook-form";
import { useContext, useState } from "react";
import { Spinner } from "./Spinner";
import { AUTHOR_TYPES } from "../types";
import { MessagesContext } from "../context";

export function InputForm() {
  const [isLoading, setIsLoading] = useState(false);
  const [, setMessages] = useContext(MessagesContext);
  const { register, handleSubmit, reset } = useForm({
    defaultValues: { query: "" },
  });
  const onSubmit = async (data) => {
    setMessages((messages) => [
      ...messages,
      { author: AUTHOR_TYPES.USER, body: query },
    ]);
    setIsLoading(true);
    reset();

    const { query } = data;
    const response = await fetch("http://127.0.0.1:5000/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ query }),
    });
    const responseData = await response.json();

    setMessages((messages) => [
      ...messages,
      { author: AUTHOR_TYPES.BOT, body: responseData["chat_gpt_answer"] },
    ]);
    setIsLoading(false);
  };

  return (
    <form className="w-full px-5 my-5" onSubmit={handleSubmit(onSubmit)}>
      {isLoading ? (
        <Spinner />
      ) : (
        <input
          {...register("query")}
          className="border w-full self-end rounded h-10 px-3 bg-white shadow"
          placeholder="Escribe tu pregunta..."
        />
      )}
    </form>
  );
}
