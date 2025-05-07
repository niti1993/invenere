import React, { useState, useEffect } from 'react';
import { Search, Book, FileText, Settings, Info, Clock, Star, ExternalLink } from 'lucide-react';

interface SearchResult {
  title: string;
  description: string;
  url: string;
  context: string;
  relevance: number;
}

function App() {
  const [searchQuery, setSearchQuery] = useState('');
  const [context, setContext] = useState<string[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [showResults, setShowResults] = useState(false);
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);

  // Simulate context detection
  useEffect(() => {
    setContext([
      'React Documentation',
      'TypeScript Guidelines',
      'Current Project Files'
    ]);
  }, []);

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    setIsSearching(true);
    // Simulate search delay and results
    setTimeout(() => {
      setSearchResults([
        {
          title: "React Hooks Documentation",
          description: "Comprehensive guide to React Hooks including useState, useEffect, and custom hooks implementation. Learn best practices and common patterns.",
          url: "https://react.dev/reference/react",
          context: "React Documentation",
          relevance: 0.95
        },
        {
          title: "TypeScript Interface Guidelines",
          description: "Learn how to properly define and use TypeScript interfaces. Includes examples of generic types and utility types.",
          url: "https://www.typescript.org/docs",
          context: "TypeScript Guidelines",
          relevance: 0.88
        },
        {
          title: "Project Configuration",
          description: "Current project setup including Vite configuration, ESLint rules, and TypeScript compiler options.",
          url: "/config",
          context: "Current Project Files",
          relevance: 0.82
        }
      ]);
      setIsSearching(false);
      setShowResults(true);
    }, 1000);
  };

  return (
    <div className="min-h-screen bg-white">
      {/* Header */}
      <header className="bg-[#5D3FD3] text-white py-4 px-6 shadow-lg">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <div className="bg-white bg-opacity-20 rounded-full p-1.5">
              <Info className="w-6 h-6" />
            </div>
            <h1 className="text-2xl font-bold">invenere</h1>
          </div>
          <button className="p-2 hover:bg-[#4c32ab] rounded-full transition-colors">
            <Settings className="w-6 h-6" />
          </button>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-6 py-12">
        {/* Search Form */}
        <form onSubmit={handleSearch} className="max-w-3xl mx-auto">
          <div className="relative">
            <Search className="absolute left-4 top-1/2 transform -translate-y-1/2 text-gray-400 w-5 h-5" />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search with context awareness..."
              className="w-full pl-12 pr-4 py-4 rounded-xl border-2 border-[#5D3FD3] focus:outline-none focus:ring-2 focus:ring-[#5D3FD3] focus:ring-opacity-50 text-lg"
            />
          </div>
        </form>

        {!showResults && (
          /* Context Section */
          <div className="mt-8 max-w-3xl mx-auto">
            <div className="flex items-center space-x-2 mb-4">
              <Book className="w-5 h-5 text-[#5D3FD3]" />
              <h2 className="text-xl font-semibold text-gray-800">Active Context</h2>
            </div>
            <div className="bg-gray-50 rounded-lg p-6 border border-gray-200">
              <div className="space-y-3">
                {context.map((item, index) => (
                  <div key={index} className="flex items-center space-x-3">
                    <FileText className="w-4 h-4 text-[#5D3FD3]" />
                    <span className="text-gray-700">{item}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Search Status */}
        {isSearching && (
          <div className="mt-8 max-w-3xl mx-auto text-center">
            <div className="inline-block px-6 py-3 bg-[#5D3FD3] bg-opacity-10 rounded-full">
              <span className="text-[#5D3FD3] font-medium">Searching with context...</span>
            </div>
          </div>
        )}

        {/* Search Results */}
        {showResults && !isSearching && (
          <div className="mt-8 max-w-3xl mx-auto">
            <div className="flex items-center justify-between mb-6">
              <div className="flex items-center space-x-2">
                <Clock className="w-5 h-5 text-[#5D3FD3]" />
                <h2 className="text-xl font-semibold text-gray-800">Search Results</h2>
              </div>
              <button
                onClick={() => setShowResults(false)}
                className="text-sm text-[#5D3FD3] hover:underline"
              >
                Back to Context
              </button>
            </div>
            <div className="space-y-6">
              {searchResults.map((result, index) => (
                <div
                  key={index}
                  className="bg-white rounded-lg p-6 border border-gray-200 hover:border-[#5D3FD3] transition-colors"
                >
                  <div className="flex items-start justify-between">
                    <div>
                      <h3 className="text-lg font-semibold text-gray-900 mb-2">
                        {result.title}
                      </h3>
                      <p className="text-gray-600 mb-3">{result.description}</p>
                      <div className="flex items-center space-x-4">
                        <span className="inline-flex items-center space-x-1 text-sm text-gray-500">
                          <FileText className="w-4 h-4" />
                          <span>{result.context}</span>
                        </span>
                        <span className="inline-flex items-center space-x-1 text-sm text-[#5D3FD3]">
                          <Star className="w-4 h-4" />
                          <span>{(result.relevance * 100).toFixed(0)}% relevant</span>
                        </span>
                      </div>
                    </div>
                    <a
                      href={result.url}
                      className="flex items-center justify-center w-8 h-8 rounded-full hover:bg-[#5D3FD3] hover:bg-opacity-10 transition-colors"
                    >
                      <ExternalLink className="w-4 h-4 text-[#5D3FD3]" />
                    </a>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;