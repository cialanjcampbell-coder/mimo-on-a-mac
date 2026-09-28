# Trial: mimo-smoke

- **When:** 2026-09-28T16:29:41 → 2026-09-28T16:35:33 (352 s)
- **Server(s):** mimo
- **Requests:** 11 (0 other/unparsed request lines)

| # | server | mode | new | reused | prefill s | gen | decode tok/s | misses/tok | finish |
|---|---|---|---|---|---|---|---|---|---|
| 1 | mimo | fresh | 1270 | 0 | 7.9 | 84 | 8.87 | 49.5 | tool_calls |
| 2 | mimo | extend | 31 | 1354 | 2.2 | 230 | 10.97 | 35.3 | tool_calls |
| 3 | mimo | extend | 141 | 1615 | 4.2 | 177 | 10.38 | 39.1 | tool_calls |
| 4 | mimo | extend | 166 | 1933 | 3.8 | 360 | 10.95 | 34.8 | tool_calls |
| 5 | mimo | extend | 232 | 2459 | 4.8 | 313 | 10.4 | 37.2 | tool_calls |
| 6 | mimo | extend | 724 | 3004 | 7.0 | 241 | 9.61 | 41.1 | tool_calls |
| 7 | mimo | extend | 259 | 3969 | 5.6 | 295 | 10.42 | 37.7 | tool_calls |
| 8 | mimo | extend | 58 | 4523 | 3.0 | 601 | 11.15 | 33.5 | tool_calls |
| 9 | mimo | extend | 57 | 5182 | 3.1 | 393 | 10.03 | 35.4 | tool_calls |
| 10 | mimo | extend | 281 | 5632 | 4.6 | 153 | 10.6 | 36.2 | tool_calls |
| 11 | mimo | extend | 147 | 6066 | 4.8 | 324 | 11.16 | 32.9 | stop |

## Totals
- Generated tokens: 3171
- Decode (token-weighted): 10.56 tok/s
- First-token proxy (prefill s): median 4.6, max 7.9

## Power and thermals (macmon)
- Samples: 347 over 348.0 s (1 glitch sample(s) >400 W dropped)
- System power: mean 60.9 W, peak 77.17 W
- CPU+GPU+ANE power: mean 11.26 W, peak 30.62 W
- GPU power: mean 11.26 W, peak 30.62 W
- Energy: 21191.5 J (5.887 Wh); per generated token: 6.683 J
- GPU temp: mean 68.39 °C, peak 77.24 °C; CPU temp: mean 65.12 °C, peak 73.71 °C
- Fan peak: 35.3% of max
- GPU busy samples: 301, median 1560 MHz, peak 1602 MHz, below 90% of peak: 9%
- RAM peak: 120.6 GB; swap change: 0.0 GB
