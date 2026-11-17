/* cardfind.js - dependency-free trading card detector + perspective warp.
   window.CardFind = { findCard(canvas) -> {corners[TL,TR,BR,BL], score, method} | null,
                       warp(canvas, corners, outW=630, outH=880) -> canvas }
   Corners are in source-canvas pixel coordinates (continuous; pixel i spans [i,i+1)).
   findCard: downscale to 360px -> gray(+saturation) -> blur -> Sobel -> oriented Hough
   (near-vertical / near-horizontal families, +-16deg) -> enumerate 2+2 line combos ->
   validate aspect/area -> score by perimeter edge support, outside quietness and
   inside/outside contrast -> prefer the largest near-best quad -> outward search for a
   weak outer edge -> robust least-squares refinement of each side. Test: tools/cardfind-test.html */
(function () {
  'use strict';
  var WORK_W = 360;
  var ASPECT = 63 / 88;
  var TH_RANGE = 16, TH_STEP = 1;               // degrees around 0 / 90
  var NTH = Math.round(2 * TH_RANGE / TH_STEP) + 1;
  var D2R = Math.PI / 180, DBG = [], REF_OUT = 5;

  function makeCanvas(w, h) {
    var c = document.createElement('canvas'); c.width = w; c.height = h; return c;
  }

  // ---- preprocessing ----
  function prep(src) {
    var sw = src.width, sh = src.height;
    var s = Math.min(1, WORK_W / sw);
    var w = Math.max(8, Math.round(sw * s)), h = Math.max(8, Math.round(sh * s));
    var c = makeCanvas(w, h), ctx = c.getContext('2d');
    ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = 'low';
    ctx.drawImage(src, 0, 0, w, h);
    var d = ctx.getImageData(0, 0, w, h).data;
    var n = w * h, g = new Float32Array(n), t = new Float32Array(n);
    // use max-contrast channel combo: luminance plus saturation-ish term (yellow borders on white)
    for (var i = 0, j = 0; i < n; i++, j += 4) {
      var r = d[j], gg = d[j + 1], b = d[j + 2];
      var mx = r > gg ? (r > b ? r : b) : (gg > b ? gg : b);
      var mn = r < gg ? (r < b ? r : b) : (gg < b ? gg : b);
      g[i] = 0.299 * r + 0.587 * gg + 0.114 * b + 0.5 * (mx - mn);
    }
    // separable [1 2 1] blur, twice
    for (var pass = 0; pass < 2; pass++) {
      for (var y = 0; y < h; y++) {
        var o = y * w;
        t[o] = g[o]; t[o + w - 1] = g[o + w - 1];
        for (var x = 1; x < w - 1; x++) t[o + x] = (g[o + x - 1] + 2 * g[o + x] + g[o + x + 1]) * 0.25;
      }
      for (y = 1; y < h - 1; y++) {
        o = y * w;
        for (x = 0; x < w; x++) g[o + x] = (t[o + x - w] + 2 * t[o + x] + t[o + x + w]) * 0.25;
      }
    }
    return { w: w, h: h, g: g, scale: s };
  }

  function sobel(P) {
    var w = P.w, h = P.h, g = P.g, n = w * h;
    var gx = new Float32Array(n), gy = new Float32Array(n), mag = new Float32Array(n);
    for (var y = 1; y < h - 1; y++) {
      for (var x = 1; x < w - 1; x++) {
        var i = y * w + x;
        var a = g[i - w - 1], b = g[i - w], c = g[i - w + 1], d = g[i - 1], f = g[i + 1],
            p = g[i + w - 1], q = g[i + w], r = g[i + w + 1];
        var sx = (c + 2 * f + r) - (a + 2 * d + p);
        var sy = (p + 2 * q + r) - (a + 2 * b + c);
        gx[i] = sx; gy[i] = sy; mag[i] = Math.sqrt(sx * sx + sy * sy);
      }
    }
    // threshold at the 85th percentile (min absolute floor)
    var hist = new Uint32Array(256), maxM = 1;
    for (i = 0; i < n; i++) if (mag[i] > maxM) maxM = mag[i];
    var k = 255 / maxM;
    for (i = 0; i < n; i++) hist[(mag[i] * k) | 0]++;
    var target = n * 0.85, acc = 0, bin = 0;
    for (; bin < 256; bin++) { acc += hist[bin]; if (acc >= target) break; }
    var thr = Math.max(bin / k, 24);
    // dilated oriented edge maps for fast perimeter scoring: ev = vertical edges, eh = horizontal
    var ev = new Uint8Array(n), eh = new Uint8Array(n), t7 = thr * 0.7;
    for (i = w; i < n - w; i++) {
      var m = mag[i]; if (m < t7) continue;
      var ax = Math.abs(gx[i]), ay = Math.abs(gy[i]);
      if (ax > 0.8 * m) { ev[i - 1] = ev[i] = ev[i + 1] = 1; }
      if (ay > 0.8 * m) { eh[i - w] = eh[i] = eh[i + w] = 1; }
    }
    return { gx: gx, gy: gy, mag: mag, thr: thr, ev: ev, eh: eh };
  }

  // ---- oriented Hough ----
  // family 0: near-vertical lines (normal angle ~0deg), family 1: near-horizontal (~90deg)
  function hough(P, E) {
    var w = P.w, h = P.h, diag = Math.ceil(Math.sqrt(w * w + h * h));
    var nR = 2 * diag + 1, acc = [new Float32Array(NTH * nR), new Float32Array(NTH * nR)];
    var cs = [[], []], sn = [[], []];
    for (var f = 0; f < 2; f++) for (var t = 0; t < NTH; t++) {
      var th = (f * 90 - TH_RANGE + t * TH_STEP) * D2R;
      cs[f].push(Math.cos(th)); sn[f].push(Math.sin(th));
    }
    var mag = E.mag, gx = E.gx, gy = E.gy, thr = E.thr, WIN = 5 / TH_STEP; // +-5 deg vote window
    for (var y = 2; y < h - 2; y++) for (var x = 2; x < w - 2; x++) {
      var i = y * w + x, m = mag[i];
      if (m < thr) continue;
      var ang = Math.atan2(gy[i], gx[i]) / D2R;      // -180..180
      if (ang < -90) ang += 180; else if (ang >= 90) ang -= 180;  // -90..90
      var fam, rel;
      if (Math.abs(ang) <= TH_RANGE + 6) { fam = 0; rel = ang; }
      else { fam = 1; rel = ang < 0 ? ang + 180 - 90 : ang - 90; if (Math.abs(rel) > TH_RANGE + 6) continue; }
      var c0 = Math.round((rel + TH_RANGE) / TH_STEP);
      var t0 = Math.max(0, c0 - WIN), t1 = Math.min(NTH - 1, c0 + WIN);
      var wgt = Math.min(m, thr * 2), A = acc[fam], C = cs[fam], S = sn[fam];
      for (t = t0; t <= t1; t++) {
        var r = Math.round(x * C[t] + y * S[t]) + diag;
        A[t * nR + r] += wgt;
      }
    }
    var out = [];
    for (f = 0; f < 2; f++) out.push(peaks(acc[f], nR, diag, f, 20));
    return out;
  }

  function peaks(A, nR, diag, fam, maxN) {
    var res = [], cand = [];
    for (var t = 0; t < NTH; t++) for (var r = 1; r < nR - 1; r++) {
      var v = A[t * nR + r];
      if (v > 0 && v >= A[t * nR + r - 1] && v >= A[t * nR + r + 1]) cand.push(t * nR + r);
    }
    cand.sort(function (a, b) { return A[b] - A[a]; });
    for (var k = 0; k < cand.length && res.length < maxN; k++) {
      var tt = (cand[k] / nR) | 0, rr = cand[k] % nR, ok = true;
      for (var q = 0; q < res.length; q++) {
        if (Math.abs(res[q].t - tt) < 4 / TH_STEP && Math.abs(res[q].r - rr) < 6) { ok = false; break; }
      }
      if (!ok) continue;
      var th = (fam * 90 - TH_RANGE + tt * TH_STEP) * D2R;
      res.push({ t: tt, r: rr, c: Math.cos(th), s: Math.sin(th), rho: rr - diag, v: A[cand[k]] });
    }
    return res;
  }

  function inter(L1, L2) {
    var det = L1.c * L2.s - L1.s * L2.c;
    if (Math.abs(det) < 1e-6) return null;
    return { x: (L1.rho * L2.s - L1.s * L2.rho) / det, y: (L1.c * L2.rho - L1.rho * L2.c) / det };
  }
  function dist(a, b) { var dx = a.x - b.x, dy = a.y - b.y; return Math.sqrt(dx * dx + dy * dy); }

  // fraction of perimeter samples with strong, correctly-oriented gradient
  function sideSupport(E, w, h, a, b, tf, cmin) {
    tf = tf || 0.7; cmin = cmin || 0.85;
    var len = dist(a, b), n = Math.max(8, Math.round(len / 2)), hit = 0, tot = 0;
    var ux = (b.x - a.x) / len, uy = (b.y - a.y) / len, nx = -uy, ny = ux;
    for (var k = 1; k < n; k++) {
      var f = k / n, px = a.x + (b.x - a.x) * f, py = a.y + (b.y - a.y) * f, best = 0;
      tot++;
      for (var o = -1; o <= 1; o++) {
        var x = Math.round(px + nx * o), y = Math.round(py + ny * o);
        if (x < 1 || y < 1 || x >= w - 1 || y >= h - 1) continue;
        var i = y * w + x, m = E.mag[i];
        if (m < E.thr * tf) continue;
        var cosv = Math.abs(E.gx[i] * nx + E.gy[i] * ny) / (m + 1e-6);
        if (cosv > cmin) { best = 1; break; }
      }
      hit += best;
    }
    return tot ? hit / tot : 0;
  }

  function fastSupport(map, w, h, a, b, step) {
    var len = dist(a, b), n = Math.max(8, Math.round(len / (step || 2))), hit = 0, tot = 0;
    var dx = (b.x - a.x) / n, dy = (b.y - a.y) / n, x = a.x + 0.5, y = a.y + 0.5;
    for (var k = 1; k < n; k++) {
      x += dx; y += dy;
      var xi = x | 0, yi = y | 0;
      tot++;
      if (xi >= 0 && yi >= 0 && xi < w && yi < h) hit += map[yi * w + xi];
    }
    return hit / tot;
  }

  // mean intensity along a segment shifted by (ox,oy)
  function stripMean(P, A, B, ox, oy) {
    var n = Math.max(6, Math.round(dist(A, B) / 3)), sum = 0, cnt = 0, w = P.w, h = P.h;
    for (var k = 1; k < n; k++) {
      var f = k / n, x = (A.x + (B.x - A.x) * f + ox) | 0, y = (A.y + (B.y - A.y) * f + oy) | 0;
      if (x >= 0 && y >= 0 && x < w && y < h) { sum += P.g[y * w + x]; cnt++; }
    }
    return cnt ? sum / cnt : 0;
  }

  function evalQuad(E, P, V1, V2, H1, H2) {
    var tl = inter(V1, H1), tr = inter(V2, H1), br = inter(V2, H2), bl = inter(V1, H2);
    if (!tl || !tr || !br || !bl) return null;
    var w = P.w, h = P.h, mx = w * 0.06, my = h * 0.06, pts = [tl, tr, br, bl];
    for (var i = 0; i < 4; i++) if (pts[i].x < -mx || pts[i].y < -my || pts[i].x > w + mx || pts[i].y > h + my) return null;
    var wt = dist(tl, tr), wb = dist(bl, br), hl = dist(tl, bl), hr = dist(tr, br);
    var cw = (wt + wb) / 2, ch = (hl + hr) / 2, asp = cw / ch;
    if (asp < 0.6 || asp > 0.84) return null;
    if (Math.min(wt, wb) / Math.max(wt, wb) < 0.7 || Math.min(hl, hr) / Math.max(hl, hr) < 0.7) return null;
    var area = 0.5 * Math.abs((tl.x * tr.y - tr.x * tl.y) + (tr.x * br.y - br.x * tr.y) + (br.x * bl.y - bl.x * br.y) + (bl.x * tl.y - tl.x * bl.y));
    var af = area / (w * h);
    if (af < 0.12 || af > 0.97) return null;
    if (cw < w * 0.3) return null;
    var aspF = 1 - Math.min(0.5, Math.abs(asp - ASPECT) / ASPECT * 1.5);
    var cheap = (fastSupport(E.eh, w, h, tl, tr, 4) + fastSupport(E.ev, w, h, tr, br, 4) + fastSupport(E.eh, w, h, br, bl, 4) + fastSupport(E.ev, w, h, bl, tl, 4)) * aspF;
    return { pts: pts, cheap: cheap, aspF: aspF, cw: cw, area: af, lines: [H1, V2, H2, V1] };
  }

  // full score: perimeter support minus parallel outside support, plus inside/outside contrast
  function fullScore(E, P, q) {
    var w = P.w, h = P.h, s = [], con = 0, P4 = q.pts, off = Math.max(4, q.cw * 0.035);
    for (var i = 0; i < 4; i++) {
      var A = P4[i], B = P4[(i + 1) % 4], map = (i & 1) ? E.ev : E.eh, L = dist(A, B);
      var ox = (B.y - A.y) / L * off, oy = -(B.x - A.x) / L * off;   // outward offset
      var on = fastSupport(map, w, h, A, B);
      var out = fastSupport(map, w, h, { x: A.x + ox, y: A.y + oy }, { x: B.x + ox, y: B.y + oy });
      s.push(Math.max(0, on - 0.6 * out));   // a silhouette edge has a quieter outside
      // a silhouette separates two differently coloured regions (texture lines do not)
      var ux = ox / off, uy = oy / off, inn = stripMean(P, A, B, -ux * 4, -uy * 4), ext = stripMean(P, A, B, ux * 4, uy * 4);
      con += Math.min(1, Math.abs(inn - ext) / 25) / 4;
    }
    var mn = Math.min(s[0], s[1], s[2], s[3]), mean = (s[0] + s[1] + s[2] + s[3]) / 4;
    q.score = (0.6 * mean + 0.4 * mn) * q.aspF * (0.7 + 0.3 * con);
  }

  // robust least-squares refinement of a side line: per sample along the side take the
  // OUTERMOST oriented gradient local-max (card silhouette, not inner frame lines)
  function refineLine(E, P, a, b, tf) {
    tf = tf || 0.45;
    var w = P.w, h = P.h, len = dist(a, b), mag = E.mag, thr = E.thr * tf;
    var ux = (b.x - a.x) / len, uy = (b.y - a.y) / len, nx = -uy, ny = ux;   // n points inward
    var T = [], D = [], M = [];
    function mAt(px, py) {
      var x = Math.round(px), y = Math.round(py);
      if (x < 1 || y < 1 || x >= w - 1 || y >= h - 1) return -1;
      var i = y * w + x, m = mag[i];
      return Math.abs(E.gx[i] * nx + E.gy[i] * ny) > 0.9 * m ? m : 0;
    }
    for (var k = Math.round(len * 0.1); k < len * 0.9; k++) {
      var px = a.x + ux * k, py = a.y + uy * k;
      for (var o = -REF_OUT; o <= 3; o++) {
        var m0 = mAt(px + nx * o, py + ny * o);
        if (m0 < thr) continue;
        var mp = mAt(px + nx * (o - 1), py + ny * (o - 1)), mn = mAt(px + nx * (o + 1), py + ny * (o + 1));
        if (m0 < mp || m0 < mn) continue;
        var den = mp - 2 * m0 + mn, off = den < 0 ? 0.5 * (mp - mn) / den : 0;
        T.push(k); D.push(o + Math.max(-0.5, Math.min(0.5, off))); M.push(m0);
        break;
      }
    }
    if (T.length < len * 0.3) return null;
    var slope = 0, icpt = 0;
    for (var it = 0; it < 3; it++) {
      var sx = 0, sy = 0, sxx = 0, sxy = 0, sw = 0;
      for (var j = 0; j < T.length; j++) {
        if (it && Math.abs(D[j] - (slope * T[j] + icpt)) > (it === 1 ? 2 : 1)) continue;
        var wt = M[j]; sx += T[j] * wt; sy += D[j] * wt; sxx += T[j] * T[j] * wt; sxy += T[j] * D[j] * wt; sw += wt;
      }
      var dd = sw * sxx - sx * sx;
      if (!sw || Math.abs(dd) < 1e-6) return null;
      slope = (sw * sxy - sx * sy) / dd; icpt = (sy - slope * sx) / sw;
    }
    if (Math.abs(slope) > 0.06 || Math.abs(icpt + slope * len / 2) > REF_OUT) return null;
    var p0 = { x: a.x + nx * icpt, y: a.y + ny * icpt };
    var p1 = { x: a.x + ux * len + nx * (icpt + slope * len), y: a.y + uy * len + ny * (icpt + slope * len) };
    var dx = p1.x - p0.x, dy = p1.y - p0.y, L = Math.sqrt(dx * dx + dy * dy);
    var c = -dy / L, s = dx / L;
    return { c: c, s: s, rho: c * p0.x + s * p0.y };
  }

  function findCard(canvas, opts) {
    opts = opts || {}; REF_OUT = opts.refOut || 5;
    var T0 = performance.now(), P = prep(canvas), T1 = performance.now(), E = sobel(P), T2 = performance.now(), L = hough(P, E), T3 = performance.now(); DBG = []; window.CardFind._dbg = DBG; DBG.t = [T1 - T0, T2 - T1, T3 - T2];
    var V = L[0], H = L[1], best = null, cands = [];
    for (var a = 0; a < V.length; a++) for (var b = 0; b < V.length; b++) {
      if (a === b) continue;
      // V1 must be left of V2 at the image middle
      var ym = P.h / 2;
      var xa = (V[a].rho - V[a].s * ym) / V[a].c, xb = (V[b].rho - V[b].s * ym) / V[b].c;
      if (xb - xa < P.w * 0.3) continue;
      for (var c = 0; c < H.length; c++) for (var d = 0; d < H.length; d++) {
        if (c === d) continue;
        var xm = P.w / 2;
        var yc = (H[c].rho - H[c].c * xm) / H[c].s, yd = (H[d].rho - H[d].c * xm) / H[d].s;
        if (yd - yc < P.h * 0.25) continue;
        var q = evalQuad(E, P, V[a], V[b], H[c], H[d]);
        if (q) cands.push(q);
      }
    }
    DBG.t.push(performance.now() - T3, cands.length);
    if (!cands.length) return null;
    cands.sort(function (x, y) { return y.cheap - x.cheap; });
    if (cands.length > 300) cands.length = 300;
    for (var i = 0; i < cands.length; i++) fullScore(E, P, cands[i]);
    var top = 0;
    for (i = 0; i < cands.length; i++) if (cands[i].score > top) top = cands[i].score;
    // among near-best candidates prefer the largest (outer card edge vs inner border edge)
    for (i = 0; i < cands.length; i++) {
      var q2 = cands[i];
      if (q2.score >= top * 0.88 && (!best || q2.area > best.area * 1.01 || (q2.area > best.area * 0.99 && q2.score > best.score))) best = q2;
    }
    if (best.score < 0.3) return null;
    // refine each side, recompute corners
    var pts = best.pts, sides = [];
    DBG.raw = best.pts.map(function (p) { return { x: (p.x + 0.5) / P.scale, y: (p.y + 0.5) / P.scale }; });
    // outward search: weak outer card edge (e.g. silver border on white) beyond the chosen line
    var cwd = (dist(pts[0], pts[1]) + dist(pts[3], pts[2])) / 2, maxOff = Math.round(cwd * 0.065);
    for (i = 0; i < 4; i++) {
      var A = pts[i], B = pts[(i + 1) % 4], len = dist(A, B);
      var ux = (B.x - A.x) / len, uy = (B.y - A.y) / len, onx = uy, ony = -ux;
      var a1 = { x: A.x + ux * len * 0.12, y: A.y + uy * len * 0.12 }, b1 = { x: B.x - ux * len * 0.12, y: B.y - uy * len * 0.12 };
      var sup = [], pick = 0;
      for (var o = 0; o <= 3 * maxOff + 1; o++)
        sup.push(sideSupport(E, P.w, P.h, { x: a1.x + onx * o, y: a1.y + ony * o }, { x: b1.x + onx * o, y: b1.y + ony * o }, 0.3, 0.96));
      for (o = 3; o <= maxOff; o++) if (sup[o] >= 0.65 && sup[o] >= sup[o - 1] && sup[o] >= sup[o + 1]) {
        // must be the outermost structure: region beyond it is quiet (rejects wood grain, clutter)
        var quiet = true;
        for (var o2 = o + 3; o2 <= o + 2 * maxOff; o2++) if (sup[o2] > 0.2) { quiet = false; break; }
        // the band between the two lines (card border) must differ from what lies beyond (background)
        if (quiet && !opts.noExpand) {
          var band = stripMean(P, a1, b1, onx * o * 0.5, ony * o * 0.5), beyond = stripMean(P, a1, b1, onx * (o + 3), ony * (o + 3));
          if (Math.abs(band - beyond) > 12) pick = o;
        }
      }
      DBG.push(pick);
      var tf = pick ? 0.3 : 0.45;
      var A2 = { x: A.x + onx * pick, y: A.y + ony * pick }, B2 = { x: B.x + onx * pick, y: B.y + ony * pick };
      var rl = refineLine(E, P, A2, B2, tf);
      if (!rl && pick) { var ddx = B2.x - A2.x, ddy = B2.y - A2.y, LL = Math.sqrt(ddx * ddx + ddy * ddy); rl = { c: -ddy / LL, s: ddx / LL }; rl.rho = rl.c * A2.x + rl.s * A2.y; }
      sides.push(rl);
    }
    var orig = best.lines, Ls = [];
    for (i = 0; i < 4; i++) Ls.push(sides[i] || orig[i]);
    var rp = [inter(Ls[3], Ls[0]), inter(Ls[0], Ls[1]), inter(Ls[1], Ls[2]), inter(Ls[2], Ls[3])];
    for (i = 0; i < 4; i++) if (!rp[i] || dist(rp[i], pts[i]) > maxOff * 1.6 + 5) rp[i] = pts[i];
    var inv = 1 / P.scale, corners = [];
    // corners are continuous canvas coords (pixel i spans [i,i+1))
    for (i = 0; i < 4; i++) corners.push({ x: (rp[i].x + 0.5) * inv, y: (rp[i].y + 0.5) * inv });
    return { corners: corners, score: Math.min(1, best.score), method: 'hough-quad' };
  }

  // ---- homography / warp ----
  function solve(A, bv) {     // Gaussian elimination with partial pivoting, n=8
    var n = bv.length;
    for (var c = 0; c < n; c++) {
      var p = c;
      for (var r = c + 1; r < n; r++) if (Math.abs(A[r][c]) > Math.abs(A[p][c])) p = r;
      var tmp = A[c]; A[c] = A[p]; A[p] = tmp; var tb = bv[c]; bv[c] = bv[p]; bv[p] = tb;
      var pv = A[c][c]; if (Math.abs(pv) < 1e-12) return null;
      for (r = c + 1; r < n; r++) {
        var f = A[r][c] / pv; if (!f) continue;
        for (var k = c; k < n; k++) A[r][k] -= f * A[c][k];
        bv[r] -= f * bv[c];
      }
    }
    var x = new Array(n);
    for (var i = n - 1; i >= 0; i--) {
      var s = bv[i];
      for (var j = i + 1; j < n; j++) s -= A[i][j] * x[j];
      x[i] = s / A[i][i];
    }
    return x;
  }
  // homography mapping src[i] -> dst[i]
  function homography(src, dst) {
    var A = [], b = [];
    for (var i = 0; i < 4; i++) {
      var x = src[i].x, y = src[i].y, u = dst[i].x, v = dst[i].y;
      A.push([x, y, 1, 0, 0, 0, -u * x, -u * y]); b.push(u);
      A.push([0, 0, 0, x, y, 1, -v * x, -v * y]); b.push(v);
    }
    var h = solve(A, b); if (!h) return null;
    h.push(1); return h;
  }

  function warp(canvas, corners, outW, outH) {
    outW = outW || 630; outH = outH || 880;
    var out = makeCanvas(outW, outH), octx = out.getContext('2d');
    var minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (var i = 0; i < 4; i++) {
      minX = Math.min(minX, corners[i].x); maxX = Math.max(maxX, corners[i].x);
      minY = Math.min(minY, corners[i].y); maxY = Math.max(maxY, corners[i].y);
    }
    var bx = Math.max(0, Math.floor(minX) - 1), by = Math.max(0, Math.floor(minY) - 1);
    var bw = Math.min(canvas.width, Math.ceil(maxX) + 2) - bx, bh = Math.min(canvas.height, Math.ceil(maxY) + 2) - by;
    if (bw < 2 || bh < 2) return out;
    var sctx = canvas.getContext('2d', { willReadFrequently: true });
    var sd = sctx.getImageData(bx, by, bw, bh).data;
    var dst = [{ x: 0, y: 0 }, { x: outW, y: 0 }, { x: outW, y: outH }, { x: 0, y: outH }];
    var src = corners.map(function (p) { return { x: p.x - bx, y: p.y - by }; });
    var H = homography(dst, src); if (!H) return out;
    var od = octx.createImageData(outW, outH), o = od.data;
    var h0 = H[0], h1 = H[1], h2 = H[2], h3 = H[3], h4 = H[4], h5 = H[5], h6 = H[6], h7 = H[7];
    var maxx = bw - 1.001, maxy = bh - 1.001, stride = bw * 4, p = 0;
    for (var y = 0; y < outH; y++) {
      var yy = y + 0.5;
      var ax = h1 * yy + h2, ay = h4 * yy + h5, aw = h7 * yy + 1;
      for (var x = 0; x < outW; x++) {
        var xx = x + 0.5, wz = h6 * xx + aw;
        var sx = (h0 * xx + ax) / wz - 0.5, sy = (h3 * xx + ay) / wz - 0.5;
        if (sx < 0) sx = 0; else if (sx > maxx) sx = maxx;
        if (sy < 0) sy = 0; else if (sy > maxy) sy = maxy;
        var x0 = sx | 0, y0 = sy | 0, fx = sx - x0, fy = sy - y0;
        var q = y0 * stride + x0 * 4, q2 = q + stride;
        var w00 = (1 - fx) * (1 - fy), w10 = fx * (1 - fy), w01 = (1 - fx) * fy, w11 = fx * fy;
        o[p] = sd[q] * w00 + sd[q + 4] * w10 + sd[q2] * w01 + sd[q2 + 4] * w11;
        o[p + 1] = sd[q + 1] * w00 + sd[q + 5] * w10 + sd[q2 + 1] * w01 + sd[q2 + 5] * w11;
        o[p + 2] = sd[q + 2] * w00 + sd[q + 6] * w10 + sd[q2 + 2] * w01 + sd[q2 + 6] * w11;
        o[p + 3] = 255; p += 4;
      }
    }
    octx.putImageData(od, 0, 0);
    return out;
  }

  window.CardFind = { findCard: findCard, warp: warp, _homography: homography };
})();
