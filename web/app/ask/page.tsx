'use client';

import { useState } from 'react';
import Image from 'next/image';

export default function Ask() {
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState<string | null>(null);
  const [sources, setSources] = useState<Array<{ source: string; page?: number | null; score: number }>>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!question.trim()) return;
    setLoading(true);
    setError(null);
    setAnswer(null);
    setSources([]);
    try {
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000'}/ask`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ question }),
      });
      if (!response.ok) {
        const text = await response.text();
        throw new Error(`API error ${response.status}: ${text}`);
      }
      const data = await response.json();
      setAnswer(data.answer);
      setSources(data.sources ?? []);
    } catch (err: any) {
      setError(err?.message ?? 'Unexpected error');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen p-4 bg-gray-50 dark:bg-gray-900">
      <h1 className="text-2xl font-bold mb-4 text-gray-900 dark:text-gray-100">Ask Clause</h1>
      <form onSubmit={handleSubmit} className="w-full max-w-md flex flex-col gap-2">
        <textarea
          className="p-2 border rounded w-full focus:outline-none focus:ring-2 focus:ring-blue-500 dark:bg-gray-800 dark:text-gray-100"
          rows={4}
          placeholder="Enter your question..."
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          required
        />
        <button
          type="submit"
          disabled={loading}
          className="bg-blue-600 text-white py-2 px-4 rounded hover:bg-blue-700 disabled:opacity-50"
        >
          {loading ? 'Asking...' : 'Ask'}
        </button>
      </form>
      {error && <p className="mt-4 text-red-600">{error}</p>}
      {answer && (
        <div className="mt-6 w-full max-w-md bg-white dark:bg-gray-800 p-4 rounded shadow">
          <h2 className="text-lg font-semibold mb-2 text-gray-800 dark:text-gray-200">Answer</h2>
          <p className="text-gray-700 dark:text-gray-300">{answer}</p>
          {sources.length > 0 && (
            <>
              <h3 className="mt-4 text-md font-medium text-gray-800 dark:text-gray-200">Sources</h3>
              <ul className="list-disc list-inside">
                {sources.map((src, idx) => (
                  <li key={idx}>
                    {src.source}{src.page != null ? ` (page ${src.page})` : ''} – score: {src.score}
                  </li>
                ))}
              </ul>
            </>
          )}
        </div>
      )}
    </div>
  );
}
