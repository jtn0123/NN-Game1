using System;
using System.Collections.Concurrent;
using System.IO;
using System.Net.Sockets;
using System.Text;
using System.Threading;

namespace CrystalCaves.Pilot
{
    // Networking stays off Unity's render thread. At most one command is in flight.
    public sealed class CaveConnection : IDisposable
    {
        readonly ConcurrentQueue<string> outgoing = new ConcurrentQueue<string>();
        readonly ConcurrentQueue<string> incoming = new ConcurrentQueue<string>();
        readonly Thread worker;
        readonly int port;
        volatile bool running = true;
        volatile bool connected;
        volatile string status = "Connecting to the local cave";
        int outstanding;
        TcpClient active;

        public bool Connected => connected;
        public bool Busy => Volatile.Read(ref outstanding) != 0;
        public string Status => status;

        public CaveConnection(int port)
        {
            this.port = port;
            worker = new Thread(Run) { IsBackground = true, Name = "Crystal Caves bridge" };
            worker.Start();
        }

        public bool Send(string json)
        {
            if (!connected || Interlocked.CompareExchange(ref outstanding, 1, 0) != 0) return false;
            outgoing.Enqueue(json);
            return true;
        }

        public bool Read(out string json) => incoming.TryDequeue(out json);

        void Run()
        {
            while (running)
            {
                try
                {
                    using (var client = new TcpClient { NoDelay = true })
                    {
                        active = client;
                        var connect = client.ConnectAsync("127.0.0.1", port);
                        if (!connect.Wait(1500)) throw new IOException("Connection timed out");
                        client.ReceiveTimeout = 3000;
                        client.SendTimeout = 3000;
                        using (var stream = client.GetStream())
                        using (var reader = new StreamReader(stream, Encoding.UTF8))
                        using (var writer = new StreamWriter(stream, new UTF8Encoding(false)) { AutoFlush = true })
                        {
                            Interlocked.Exchange(ref outstanding, 1);
                            writer.WriteLine("{\"op\":\"snapshot\"}");
                            Receive(reader);
                            connected = true;
                            status = "Connected";
                            while (running)
                            {
                                if (!outgoing.TryDequeue(out var message)) { Thread.Sleep(3); continue; }
                                writer.WriteLine(message);
                                Receive(reader);
                            }
                        }
                    }
                }
                catch (Exception error) when (error is IOException || error is SocketException || error is AggregateException || error is ObjectDisposedException)
                {
                    connected = false;
                    status = "Waiting for the local cave";
                    while (outgoing.TryDequeue(out _)) { }
                    Interlocked.Exchange(ref outstanding, 0);
                    if (running) Thread.Sleep(800);
                }
            }
        }

        void Receive(StreamReader reader)
        {
            var line = reader.ReadLine();
            if (line == null) throw new IOException("Bridge disconnected");
            if (line.Length > 256 * 1024) throw new IOException("Invalid bridge response");
            incoming.Enqueue(line);
            Interlocked.Exchange(ref outstanding, 0);
        }

        public void Dispose()
        {
            running = connected = false;
            active?.Close();
            worker.Join(2000);
        }
    }
}
