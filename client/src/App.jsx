import { MessagesProvider } from "./src/components/MessagesProvider";
import { MessageList } from "./src/components/MessageList";
import { InputForm } from "./src/components/InputForm";

function App() {
  return (
    <MessagesProvider>
      <div className="flex flex-col h-screen bg-white">
        <MessageList />
        <InputForm />
      </div>
    </MessagesProvider>
  );
}

export default App;
