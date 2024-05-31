import { useForm } from "react-hook-form";
import { useContext, useState } from "react";
import { Spinner } from "./Spinner";
import { AUTHOR_TYPES } from "../types";
import { MessagesContext } from "../context";
import { sendMessage } from "../actions/sendMessage";

export function InputForm() {
  const [isLoading, setIsLoading] = useState(false);
  const [, setMessages] = useContext(MessagesContext);
  const { register, handleSubmit, reset } = useForm({
    defaultValues: { query: "" },
  });
  const onSubmit = async (data: any) => {
    try {
      const query = data.query;
      setMessages((messages) => [
        ...messages,
        { author: AUTHOR_TYPES.USER, body: query },
      ]);
      setIsLoading(true);
      reset();

      const newMessage = await sendMessage(query);
      setMessages((messages) => [...messages, newMessage]);

      // TODO: Add also a message with file name - link to the file.
    } catch (error) {
      console.error(error);
    }
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
