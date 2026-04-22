#property copyright ""
#property version   "1.00"
#property strict

input string InpSymbols    = "EURUSD,GBPUSD,USDCHF,USDCAD,XAUUSD";
input string InpTimeframes = "H4,H1,M15,M5";
input int    InpBars       = 300;
input string InpEndpoint   = "tcp://127.0.0.1:7777";
input int    InpRefreshMs  = 1000;

#import "libzmq.dll"
int  zmq_ctx_new();
int  zmq_socket(int context, int type);
int  zmq_bind(int socket, string endpoint);
int  zmq_send(int socket, const uchar &buffer[], int length, int flags);
void zmq_close(int socket);
void zmq_ctx_term(int context);
#import

#define ZMQ_PUB 1

int g_ctx = 0;
int g_sock = 0;
string g_symbols[];
ENUM_TIMEFRAMES g_tfs[];

string TimeframeToString(ENUM_TIMEFRAMES tf)
{
   switch(tf)
   {
      case PERIOD_M1:  return "m1";
      case PERIOD_M5:  return "m5";
      case PERIOD_M15: return "m15";
      case PERIOD_M30: return "m30";
      case PERIOD_H1:  return "h1";
      case PERIOD_H4:  return "h4";
      case PERIOD_D1:  return "1d";
   }
   return "";
}

ENUM_TIMEFRAMES StringToTimeframe(string tf)
{
   StringTrimLeft(tf);
   StringTrimRight(tf);
   tf = StringToUpper(tf);
   if(tf == "M1")  return PERIOD_M1;
   if(tf == "M5")  return PERIOD_M5;
   if(tf == "M15") return PERIOD_M15;
   if(tf == "M30") return PERIOD_M30;
   if(tf == "H1")  return PERIOD_H1;
   if(tf == "H4")  return PERIOD_H4;
   if(tf == "D1")  return PERIOD_D1;
   if(tf == "W1")  return PERIOD_W1;
   if(tf == "MN1") return PERIOD_MN1;
   return PERIOD_CURRENT;
}

bool SendPayload(const string payload)
{
   uchar bytes[];
   int len = StringToCharArray(payload, bytes, 0, WHOLE_ARRAY, CP_UTF8) - 1;
   if(len <= 0) return false;
   return (zmq_send(g_sock, bytes, len, 0) >= 0);
}

int OnInit()
{
   int sym_count = StringSplit(InpSymbols, ',', g_symbols);
   if(sym_count <= 0)
   {
      Print("FXProBridge: no symbols configured");
      return(INIT_FAILED);
   }

   string tf_tokens[];
   int tf_count = StringSplit(InpTimeframes, ',', tf_tokens);
   ArrayResize(g_tfs, tf_count);
   for(int i=0; i<tf_count; i++)
   {
      string token = tf_tokens[i];
      StringTrimLeft(token);
      StringTrimRight(token);
      g_tfs[i] = StringToTimeframe(token);
   }

   g_ctx = zmq_ctx_new();
   if(g_ctx == 0)
   {
      Print("FXProBridge: zmq_ctx_new failed");
      return(INIT_FAILED);
   }

   g_sock = zmq_socket(g_ctx, ZMQ_PUB);
   if(g_sock == 0)
   {
      Print("FXProBridge: zmq_socket failed");
      return(INIT_FAILED);
   }

   if(zmq_bind(g_sock, InpEndpoint) != 0)
   {
      Print("FXProBridge: zmq_bind failed ", InpEndpoint);
      return(INIT_FAILED);
   }

   EventSetTimer(InpRefreshMs / 1000.0);
   Print("FXProBridge started at ", InpEndpoint);
   return(INIT_SUCCEEDED);
}

void OnDeinit(const int reason)
{
   EventKillTimer();
   if(g_sock != 0) zmq_close(g_sock);
   if(g_ctx  != 0) zmq_ctx_term(g_ctx);
}

void OnTimer()
{
   int sym_count = ArraySize(g_symbols);
   int tf_count  = ArraySize(g_tfs);

   for(int s=0; s<sym_count; s++)
   {
      string sym = g_symbols[s];
      StringTrimLeft(sym);
      StringTrimRight(sym);
      if(sym == "") continue;

      for(int t=0; t<tf_count; t++)
      {
         ENUM_TIMEFRAMES tf = g_tfs[t];
         if(tf == PERIOD_CURRENT) continue;

         int bars = iBars(sym, tf);
         if(bars <= 0) continue;
         if(bars > InpBars) bars = InpBars;

         MqlRates rates[];
         int copied = CopyRates(sym, tf, 0, bars, rates);
         if(copied <= 0) continue;

         ArraySetAsSeries(rates, true);

         for(int i=bars-1; i>=0; i--)
         {
            MqlRates r = rates[i];
            string payload = StringFormat(
               "{\"symbol\":\"%s\",\"tf\":\"%s\",\"timestamp\":\"%s\",\"open\":%g,\"high\":%g,\"low\":%g,\"close\":%g,\"volume\":%g}",
               sym,
               TimeframeToString(tf),
               TimeToString(r.time, TIME_DATE|TIME_MINUTES|TIME_SECONDS),
               r.open, r.high, r.low, r.close, r.tick_volume
            );
            SendPayload(payload);
         }
      }
   }
}
