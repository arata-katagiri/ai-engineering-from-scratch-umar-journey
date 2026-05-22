import Anthropic from "@anthropic-ai/sdk"
import * as dotenv from "dotenv"
dotenv.config({ path: "/home/umar/Documents/learn_ai/ai-engineering-from-scratch/.env" });

const client = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY });

const response = await client.messages.create({
    model: "claude-sonnet-4-20250514",
    max_tokens: 256,
    messages: [{ role: "user", content: "What is neural network in one sentence?"}],
});

console.log(response.content[0].type == "text" ? response.content[0].text : "");