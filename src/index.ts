#!/usr/bin/env node
import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import {
  CallToolRequestSchema,
  ErrorCode,
  ListToolsRequestSchema,
  McpError,
} from '@modelcontextprotocol/sdk/types.js';
import OpenAI from 'openai';

const API_KEY = process.env.PERPLEXITY_API_KEY;
if (!API_KEY) {
  throw new Error('PERPLEXITY_API_KEY environment variable is required');
}

type SearchType = 'research' | 'troubleshoot' | 'update';
type ComplexityLevel = 'low' | 'medium' | 'high';

interface SearchArgs {
  query: string;
  type?: SearchType;
  complexity?: ComplexityLevel;
}

const isValidSearchArgs = (args: any): args is SearchArgs =>
  typeof args === 'object' &&
  args !== null &&
  typeof args.query === 'string' &&
  args.query.length > 0 &&
  (args.type === undefined || ['research', 'troubleshoot', 'update'].includes(args.type)) &&
  (args.complexity === undefined || ['low', 'medium', 'high'].includes(args.complexity));

class PerplexityServer {
  private server: Server;
  private perplexity: OpenAI;
  private readonly systemMessages: Record<SearchType, string> = {
    research: `You are a specialized AI research assistant focused on software development and technology. Your primary role is to:
- Find and analyze the latest updates, releases, and changes in software technologies
- Provide detailed technical documentation and specifications
- Compare different approaches and best practices
- Include specific version numbers, release dates, and documentation links
- Cite sources and reference official documentation when available
Always structure your responses with clear headings and code examples where relevant.`,
    troubleshoot: `You are a technical troubleshooting assistant specialized in software development. Your primary role is to:
- Analyze error messages and identify potential causes
- Suggest specific debugging steps and solutions
- Reference known issues and bug reports
- Provide workarounds when available
- Include relevant code examples and configuration snippets
Always include both immediate fixes and long-term solutions when applicable.`,
    update: `You are a technology update tracking assistant. Your primary role is to:
- Monitor and report on the latest software releases and updates
- Highlight breaking changes and deprecations
- Explain migration paths and upgrade procedures
- Compare features across versions
- Provide specific version numbers and changelog details
Always include release dates and backward compatibility information.`
  };

  private readonly modelsByComplexity: Record<ComplexityLevel, string> = {
    low: 'llama-3.1-sonar-small-128k-online',
    medium: 'llama-3.1-sonar-large-128k-online',
    high: 'llama-3.1-sonar-huge-128k-online'
  };

  constructor() {
    this.server = new Server(
      {
        name: 'perplexity-server',
        version: '0.1.0',
      },
      {
        capabilities: {
          tools: {},
        },
      }
    );

    this.perplexity = new OpenAI({
      apiKey: API_KEY,
      baseURL: 'https://api.perplexity.ai',
    });

    this.setupToolHandlers();
    
    // Error handling
    this.server.onerror = (error) => console.error('[MCP Error]', error);
    process.on('SIGINT', async () => {
      await this.server.close();
      process.exit(0);
    });
  }

  private getSystemMessage(type: SearchType = 'research'): string {
    return this.systemMessages[type];
  }

  private getModelForComplexity(complexity: ComplexityLevel = 'medium'): string {
    return this.modelsByComplexity[complexity];
  }

  private async tryWithEscalation(query: string, type: SearchType, initialComplexity: ComplexityLevel): Promise<OpenAI.Chat.Completions.ChatCompletion> {
    const complexityLevels: ComplexityLevel[] = ['low', 'medium', 'high'];
    const startIndex = complexityLevels.indexOf(initialComplexity);
    
    for (let i = startIndex; i < complexityLevels.length; i++) {
      try {
        const response = await this.perplexity.chat.completions.create({
          model: this.getModelForComplexity(complexityLevels[i] as ComplexityLevel),
          messages: [
            { 
              role: 'system', 
              content: this.getSystemMessage(type)
            },
            { 
              role: 'user', 
              content: query 
            }
          ],
          max_tokens: 4000
        });

        if (response.choices && response.choices.length > 0 && response.choices[0].message?.content) {
          const content = response.choices[0].message.content;
          // Check if response seems incomplete or too basic
          if (i < complexityLevels.length - 1 && 
              (content.length < 200 || !content.includes('\n') || content.includes("I apologize") || 
               content.includes("I'm not sure"))) {
            continue; // Escalate to next model
          }
          return response;
        }
      } catch (error) {
        if (i === complexityLevels.length - 1) throw error;
        console.error(`Error with ${complexityLevels[i]} model, escalating:`, error);
      }
    }
    throw new Error('All model attempts failed');
  }

  private setupToolHandlers() {
    this.server.setRequestHandler(ListToolsRequestSchema, async () => ({
      tools: [
        {
          name: 'search_web',
          description: 'Research technical topics, troubleshoot issues, or get latest updates using Perplexity AI',
          inputSchema: {
            type: 'object',
            properties: {
              query: {
                type: 'string',
                description: 'The search query',
              },
              type: {
                type: 'string',
                enum: ['research', 'troubleshoot', 'update'],
                description: 'Type of search (research for in-depth analysis, troubleshoot for problem-solving, update for latest changes)',
              },
              complexity: {
                type: 'string',
                enum: ['low', 'medium', 'high'],
                description: 'Expected complexity of the query',
              },
            },
            required: ['query'],
          },
        },
      ],
    }));

    this.server.setRequestHandler(CallToolRequestSchema, async (request) => {
      if (request.params.name !== 'search_web') {
        throw new McpError(
          ErrorCode.MethodNotFound,
          `Unknown tool: ${request.params.name}`
        );
      }

      if (!isValidSearchArgs(request.params.arguments)) {
        throw new McpError(
          ErrorCode.InvalidParams,
          'Invalid search arguments. Must provide a query string and optional type/complexity.'
        );
      }

      try {
        const { query, type = 'research', complexity = 'medium' } = request.params.arguments;
        const response = await this.tryWithEscalation(
          query, 
          type as SearchType, 
          complexity as ComplexityLevel
        );

        const content = response.choices[0].message?.content;
        if (!content) {
          throw new Error('No response content received');
        }

        return {
          content: [
            {
              type: 'text',
              text: content
            },
          ],
        };
      } catch (error) {
        console.error('API Error:', error);
        return {
          content: [
            {
              type: 'text',
              text: `Perplexity API error: ${
                error instanceof Error ? error.message : String(error)
              }`,
            },
          ],
          isError: true,
        };
      }
    });
  }

  async run() {
    const transport = new StdioServerTransport();
    await this.server.connect(transport);
    console.error('Perplexity MCP server running on stdio');
  }
}

const server = new PerplexityServer();
server.run().catch(console.error);
