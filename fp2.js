/* fp2.js - visual fingerprint v2 for Pokemon card images (mirror of tools/build_fp2.py).
 *
 * window.FP2 = {
 *   RECORD_BYTES,                       // 396 bytes per DB record (int8 stored as raw bytes)
 *   fingerprint(canvasOrImage) -> Int8Array  // upright card image, any size (resized to 630x880
 *        with a Pillow-exact bilinear resample in JS, then box-averaged).
 *        Returns VARIANTS*396 bytes: bytes [0,396) are the plain record (identical to the
 *        Python DB record of the same image); the rest are the query under 224 small
 *        geometric variants (shift +-4%, scale +-4%, rotation +-2 deg) used to absorb
 *        residual mis-alignment of the card warp.
 *   record(fp) -> Int8Array             // the plain 396-byte record of a fingerprint
 *   distance(fp, dbBytes, offset) -> number  // min over fp's variants; lower = more similar
 *   search(fp, dbBytes, k=5, n1=100) -> [{index, distance}]  // fast 2-stage full-DB scan
 * }
 * dbBytes: Uint8Array / Int8Array / ArrayBuffer of data/fp2-{en,ja}.bin; offset in bytes.
 */
(function () {
  "use strict";
  var W = 630, H = 880, BH = 110, BW = 90, GH = 22, GW = 15, CH = 11, CW = 3;
  var NG = GH * GW, NC = CH * CW * 2, REC = NG + NC;
  var G_SCALE = 24, C_SCALE = 127, C_WEIGHT = 0.5;

  // variant list - identical order to build_fp2.variant_params(); first 9 = stage-1 set
  var VARS = (function () {
    var first = [], all = [], sh = [0, -0.04, 0.04, -0.02, 0.02], a, b, c, d;
    var s9 = [0, -0.04, 0.04];
    for (a = 0; a < 3; a++) for (b = 0; b < 3; b++) first.push([s9[a], s9[b], 1, 0]);
    var key = function (v) { return v.join(","); };
    var seen = {};
    first.forEach(function (v) { seen[key(v)] = 1; all.push(v); });
    for (a = 0; a < 5; a++) for (b = 0; b < 5; b++) for (c = 0; c < 3; c++) for (d = 0; d < 3; d++) {
      var v = [sh[a], sh[b], [1, 0.96, 1.04][c], [0, -2, 2][d]];
      if (!seen[key(v)]) all.push(v);
    }
    return all;
  })();
  var NV = VARS.length; // 225

  function q(v, scale) {
    var x = Math.floor(v * scale + 0.5);
    return x < -127 ? -127 : (x > 127 ? 127 : x);
  }

  // area-average the 630x880 RGBA pixels into the 110x90 RGB base (8x7 blocks)
  function baseOf(px) {
    var b = new Float64Array(BH * BW * 3);
    for (var by = 0; by < BH; by++) {
      for (var bx = 0; bx < BW; bx++) {
        var r = 0, g = 0, bl = 0;
        for (var y = by * 8; y < by * 8 + 8; y++) {
          var p = (y * W + bx * 7) * 4;
          for (var x = 0; x < 7; x++, p += 4) { r += px[p]; g += px[p + 1]; bl += px[p + 2]; }
        }
        var o = (by * BW + bx) * 3;
        b[o] = r / 56; b[o + 1] = g / 56; b[o + 2] = bl / 56;
      }
    }
    return b;
  }

  // base (110x90x3 float) -> 396 int8 written into out at off
  var gTmp = new Float64Array(NG), aTmp = new Float64Array(NG);
  function descInto(b, out, off) {
    var gy, gx, y, x, i, j;
    for (gy = 0; gy < GH; gy++) for (gx = 0; gx < GW; gx++) {
      var s = 0;
      for (y = gy * 5; y < gy * 5 + 5; y++) for (x = gx * 6; x < gx * 6 + 6; x++) {
        i = (y * BW + x) * 3;
        s += b[i] * 0.299 + b[i + 1] * 0.587 + b[i + 2] * 0.114;
      }
      gTmp[gy * GW + gx] = s / 30;
    }
    for (gy = 0; gy < GH; gy++) for (gx = 0; gx < GW; gx++) {
      var acc = 0;
      for (var dy = -2; dy <= 2; dy++) {
        var yy = gy + dy; yy = yy < 0 ? 0 : (yy >= GH ? GH - 1 : yy);
        for (var dx = -2; dx <= 2; dx++) {
          var xx = gx + dx; xx = xx < 0 ? 0 : (xx >= GW ? GW - 1 : xx);
          acc += gTmp[yy * GW + xx];
        }
      }
      aTmp[gy * GW + gx] = gTmp[gy * GW + gx] - acc / 25;
    }
    var m = 0; for (j = 0; j < NG; j++) m += aTmp[j]; m /= NG;
    var v = 0; for (j = 0; j < NG; j++) { aTmp[j] -= m; v += aTmp[j] * aTmp[j]; }
    var sd = Math.sqrt(v / NG) + 1e-6;
    for (j = 0; j < NG; j++) out[off + j] = q(aTmp[j] / sd, G_SCALE);
    var k = off + NG;
    for (var cy = 0; cy < CH; cy++) for (var cx = 0; cx < CW; cx++) {
      var r = 0, g = 0, bl = 0;
      for (y = cy * 10; y < cy * 10 + 10; y++) for (x = cx * 30; x < cx * 30 + 30; x++) {
        i = (y * BW + x) * 3; r += b[i]; g += b[i + 1]; bl += b[i + 2];
      }
      r /= 300; g /= 300; bl /= 300;
      var ss = r + g + bl + 3;
      out[k++] = q((r - g) / ss, C_SCALE);
      out[k++] = q((r + g - 2 * bl) / (2 * ss), C_SCALE);
    }
  }

  function warp(b, p, dst) {
    var dx = p[0], dy = p[1], s = p[2], rad = p[3] * Math.PI / 180;
    var c = Math.cos(rad), sn = Math.sin(rad);
    for (var yy = 0; yy < BH; yy++) {
      var Y = (yy + 0.5) * 8 - H / 2;
      for (var xx = 0; xx < BW; xx++) {
        var X = (xx + 0.5) * 7 - W / 2;
        var sx = (s * (c * X - sn * Y) + W / 2 + dx * W) / 7 - 0.5;
        var sy = (s * (sn * X + c * Y) + H / 2 + dy * H) / 8 - 0.5;
        sx = sx < 0 ? 0 : (sx > BW - 1 ? BW - 1 : sx);
        sy = sy < 0 ? 0 : (sy > BH - 1 ? BH - 1 : sy);
        var x0 = Math.floor(sx), y0 = Math.floor(sy);
        var x1 = x0 + 1 > BW - 1 ? BW - 1 : x0 + 1, y1 = y0 + 1 > BH - 1 ? BH - 1 : y0 + 1;
        var fx = sx - x0, fy = sy - y0;
        var w00 = (1 - fx) * (1 - fy), w01 = fx * (1 - fy), w10 = (1 - fx) * fy, w11 = fx * fy;
        var i00 = (y0 * BW + x0) * 3, i01 = (y0 * BW + x1) * 3, i10 = (y1 * BW + x0) * 3, i11 = (y1 * BW + x1) * 3;
        var o = (yy * BW + xx) * 3;
        for (var ch = 0; ch < 3; ch++)
          dst[o + ch] = b[i00 + ch] * w00 + b[i01 + ch] * w01 + b[i10 + ch] * w10 + b[i11 + ch] * w11;
      }
    }
    return dst;
  }

  // Pillow-exact separable BILINEAR resample (Resample.c, 8bpc fixed point, 22 precision bits)
  var PREC = 22, ONE = 1 << PREC, HALF = 1 << (PREC - 1);
  function coeffs(inSize, outSize) {
    var scale = inSize / outSize, fs = scale < 1 ? 1 : scale, support = fs, ss = 1 / fs;
    var ksize = Math.ceil(support) * 2 + 1;
    var kk = new Int32Array(outSize * ksize), lo = new Int32Array(outSize), n = new Int32Array(outSize);
    var w = new Float64Array(ksize);
    for (var xx = 0; xx < outSize; xx++) {
      var center = (xx + 0.5) * scale, ww = 0;
      var xmin = Math.trunc(center - support + 0.5); if (xmin < 0) xmin = 0;
      var xmax = Math.trunc(center + support + 0.5); if (xmax > inSize) xmax = inSize;
      xmax -= xmin;
      for (var x = 0; x < xmax; x++) {
        var t = Math.abs((x + xmin - center + 0.5) * ss);
        w[x] = t < 1 ? 1 - t : 0; ww += w[x];
      }
      for (x = 0; x < xmax; x++) {
        var v = ww !== 0 ? w[x] / ww : w[x];
        kk[xx * ksize + x] = v < 0 ? Math.trunc(-0.5 + v * ONE) : Math.trunc(0.5 + v * ONE);
      }
      lo[xx] = xmin; n[xx] = xmax;
    }
    return { kk: kk, lo: lo, n: n, ksize: ksize };
  }
  function clip8(s) { s = Math.floor(s / ONE); return s < 0 ? 0 : (s > 255 ? 255 : s); }
  function resizePIL(px, iw, ih) { // RGBA in -> RGBA 630x880 out (alpha ignored)
    var cx = coeffs(iw, W), cy = coeffs(ih, H), tmp = new Uint8Array(W * ih * 4), out = new Uint8ClampedArray(W * H * 4);
    var y, x, k, c, s0, s1, s2, p, kv;
    for (y = 0; y < ih; y++) for (x = 0; x < W; x++) {
      s0 = s1 = s2 = HALF;
      for (k = 0; k < cx.n[x]; k++) {
        kv = cx.kk[x * cx.ksize + k]; p = (y * iw + cx.lo[x] + k) * 4;
        s0 += px[p] * kv; s1 += px[p + 1] * kv; s2 += px[p + 2] * kv;
      }
      c = (y * W + x) * 4; tmp[c] = clip8(s0); tmp[c + 1] = clip8(s1); tmp[c + 2] = clip8(s2);
    }
    for (y = 0; y < H; y++) for (x = 0; x < W; x++) {
      s0 = s1 = s2 = HALF;
      for (k = 0; k < cy.n[y]; k++) {
        kv = cy.kk[y * cy.ksize + k]; p = ((cy.lo[y] + k) * W + x) * 4;
        s0 += tmp[p] * kv; s1 += tmp[p + 1] * kv; s2 += tmp[p + 2] * kv;
      }
      c = (y * W + x) * 4; out[c] = clip8(s0); out[c + 1] = clip8(s1); out[c + 2] = clip8(s2); out[c + 3] = 255;
    }
    return out;
  }

  var cv = null;
  function pixels630(src) {
    var iw = src.naturalWidth || src.videoWidth || src.width, ih = src.naturalHeight || src.videoHeight || src.height;
    if (!cv) cv = document.createElement("canvas");
    cv.width = iw; cv.height = ih;
    var ctx = cv.getContext("2d", { willReadFrequently: true });
    ctx.fillStyle = "#fff"; ctx.fillRect(0, 0, iw, ih); // flatten alpha on white (as Python)
    ctx.drawImage(src, 0, 0, iw, ih);
    var px = ctx.getImageData(0, 0, iw, ih).data;
    return (iw === W && ih === H) ? px : resizePIL(px, iw, ih); // same resize as PIL Image.BILINEAR
  }

  function fingerprint(src) {
    var b = baseOf(pixels630(src));
    var out = new Int8Array(NV * REC), tmp = new Float64Array(BH * BW * 3);
    descInto(b, out, 0);
    for (var v = 1; v < NV; v++) descInto(warp(b, VARS[v], tmp), out, v * REC);
    return out;
  }

  var views = new WeakMap();
  function i8(db) {
    if (db instanceof Int8Array) return db;
    var buf = db instanceof ArrayBuffer ? db : db.buffer;
    var off = db instanceof ArrayBuffer ? 0 : db.byteOffset, len = db.byteLength;
    var m = views.get(buf); if (!m) { m = {}; views.set(buf, m); }
    var k = off + ":" + len;
    return m[k] || (m[k] = new Int8Array(buf, off, len));
  }

  // distance of one fp variant (at fo) to record (at ro); aborts early above `limit`
  function d1(fp, fo, db, ro, limit) {
    var s = 0, j, t;
    for (j = 0; j < NG; j++) { t = fp[fo + j] - db[ro + j]; s += t < 0 ? -t : t; }
    if (s > limit) return s;
    var c = 0;
    for (j = NG; j < REC; j++) { t = fp[fo + j] - db[ro + j]; c += t < 0 ? -t : t; }
    return s + C_WEIGHT * c;
  }

  function minDist(fp, nv, db, ro, limit) {
    var best = Infinity;
    for (var v = 0; v < nv; v++) {
      var d = d1(fp, v * REC, db, ro, Math.min(best, limit));
      if (d < best) best = d;
    }
    return best;
  }

  function distance(fp, dbBytes, offset) {
    return minDist(fp, Math.max(1, Math.floor(fp.length / REC)), i8(dbBytes), offset || 0, Infinity);
  }

  function search(fp, dbBytes, k, n1) {
    k = k || 5; n1 = n1 || 100;
    var db = i8(dbBytes), n = Math.floor(db.length / REC);
    var nv = Math.max(1, Math.floor(fp.length / REC)), nv1 = Math.min(9, nv);
    // stage 1: 9 shift variants, keep n1 best (sorted insertion; threshold = current n1-th)
    var bi = new Int32Array(n1), bd = new Float64Array(n1).fill(Infinity), thr = Infinity;
    for (var r = 0; r < n; r++) {
      var d = minDist(fp, nv1, db, r * REC, thr);
      if (d < thr) {
        var p = n1 - 1;
        while (p > 0 && bd[p - 1] > d) { bd[p] = bd[p - 1]; bi[p] = bi[p - 1]; p--; }
        bd[p] = d; bi[p] = r; thr = bd[n1 - 1];
      }
    }
    // stage 2: all variants on the shortlist
    var res = [];
    for (var i = 0; i < n1 && bd[i] < Infinity; i++)
      res.push({ index: bi[i], distance: minDist(fp, nv, db, bi[i] * REC, Infinity) });
    res.sort(function (a, b) { return a.distance - b.distance || a.index - b.index; });
    return res.slice(0, k);
  }

  window.FP2 = {
    RECORD_BYTES: REC, VARIANTS: NV,
    fingerprint: fingerprint,
    record: function (fp) { return fp.subarray(0, REC); },
    distance: distance,
    search: search
  };
})();
