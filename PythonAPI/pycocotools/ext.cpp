#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <omp.h>

namespace py = pybind11;

struct image_struct {
  size_t id;
  int h;
  int w;
};

typedef struct {
  unsigned long h;
  unsigned long w;
  unsigned long m;
  unsigned int *cnts;
} RLE;

struct anns_struct {
  int image_id;
  int category_id;
  size_t id;
  float area;
  int iscrowd;
  float score;
  std::vector<float> bbox;
  // segmentation
  std::vector<std::vector<double>> segm_list;
  std::vector<int> segm_size;
  std::vector<int> segm_counts_list;
  std::string segm_counts_str;
};

// dict type
struct data_struct {
  std::vector<float> area;
  std::vector<int> iscrowd;
  std::vector<std::vector<float>> bbox;
  std::vector<int> ignore;
  std::vector<float> score;
  std::vector<std::vector<int>> segm_size;
  std::vector<std::string> segm_counts;
  std::vector<int64_t> id;
};


// create index results
std::vector<int64_t> imgids;
std::vector<int64_t> catids;
std::unordered_map<size_t, image_struct> imgsgt;
std::vector<std::vector<double>> gtbboxes;
std::vector<std::vector<RLE>> gtsegm;
// TODO: clean
// for (size_t i = 0; i < g.size(); i++) {free(g[i].cnts);}
//std::unordered_map<size_t, image_struct> imgsdt;
//std::unordered_map<size_t, anns_struct> annsgt;
//std::unordered_map<size_t, anns_struct> annsdt;
//std::unordered_map<size_t, std::vector<anns_struct>> gtimgToAnns;
//std::unordered_map<size_t, std::vector<anns_struct>> dtimgToAnns;



// internal prepare results
inline size_t key(int i,int j) {
  return static_cast<size_t>(i) << 32 | static_cast<unsigned int>(j);
}
std::unordered_map<size_t, data_struct> gts_map;
std::unordered_map<size_t, data_struct> dts_map;
std::unordered_map<size_t, size_t> gtinds;
std::unordered_map<size_t, size_t> dtinds;

// internal computeiou results
std::unordered_map<size_t, std::vector<double>> ious_map;
// std::unordered_map<size_t, std::shared_ptr<std::vector<double>>> ious_map;


template <typename T, typename Comparator = std::greater<T> >
std::vector<size_t> sort_indices(const std::vector<T>& v, Comparator comparator = Comparator()) {
  std::vector<size_t> indices(v.size());
  std::iota(indices.begin(), indices.end(), 0);
  std::sort(indices.begin(), indices.end(),
            [&](size_t i1, size_t i2) { return comparator(v[i1], v[i2]); });
  return indices;
}

template <typename T, typename Comparator = std::greater<T> >
std::vector<size_t> stable_sort_indices(const std::vector<T>& v, Comparator comparator = Comparator()) {
  std::vector<size_t> indices(v.size());
  std::iota(indices.begin(), indices.end(), 0);
  std::stable_sort(indices.begin(), indices.end(),
                   [&](size_t i1, size_t i2) { return comparator(v[i1], v[i2]); });
  return indices;
}

void accumulate(int T, int A, const std::vector<int>& maxDets, const std::vector<double>& recThrs,
                std::vector<double>& precision,
                std::vector<double>& recall,
                std::vector<double>& scores,
                int K, int I, int R, int M, int k, int a,
                const std::vector<std::vector<int64_t>>& gtignore,
                const std::vector<std::vector<double>>& dtignore,
                const std::vector<std::vector<double>>& dtmatches,
                const std::vector<std::vector<double>>& dtscores);

void compute_iou(std::string iouType, int maxDet, int useCats);

std::tuple<py::array_t<int64_t>,py::array_t<int64_t>,py::dict>
cpp_evaluate(int useCats,
             std::vector<std::vector<double>> areaRngs,
             std::vector<double> iouThrs_ptr,
             std::vector<int> maxDets, std::vector<double> recThrs, std::string iouType, int nthreads) {
  assert(useCats > 0);

  int T = iouThrs_ptr.size();
  int A = areaRngs.size();
  int K = catids.size();
  int R = recThrs.size();
  int M = maxDets.size();
  std::vector<double> precision(T*R*K*A*M);
  std::vector<double> recall(T*K*A*M);
  std::vector<double> scores(T*R*K*A*M);

  int maxDet = maxDets[M-1];
  compute_iou(iouType, maxDet, useCats);

  #pragma omp parallel for num_threads(nthreads)
  for(size_t c = 0; c < catids.size(); c++) {
    for(size_t a = 0; a < areaRngs.size(); a++) {
      const double aRng0 = areaRngs[a][0];
      const double aRng1 = areaRngs[a][1];

      std::vector<std::vector<int64_t>> gtIgnore_list;
      std::vector<std::vector<double>> dtIgnore_list;
      std::vector<std::vector<double>> dtMatches_list;
      std::vector<std::vector<double>> dtScores_list;

      for(size_t i = 0; i < imgids.size(); i++) {
        const int catId = catids[c];
        const int imgId = imgids[i];
        auto& gtsm = gts_map[key(imgId, catId)];
        auto& dtsm = dts_map[key(imgId, catId)];
        if((gtsm.id.size()==0) && (dtsm.id.size()==0)) {
          continue;
        }

        // sizes
        const int T = iouThrs_ptr.size();
        const int G = gtsm.id.size();
        const int Do = dtsm.id.size();
        const int D = std::min(Do, maxDet);
        const int I = (G==0||D==0) ? 0 : D;

        // arrays
        std::vector<double> gtm(T*G, 0.0);
        gtIgnore_list.push_back(std::vector<int64_t>(G));
        dtIgnore_list.push_back(std::vector<double>(T*D, 0.0));
        dtMatches_list.push_back(std::vector<double>(T*D, 0.0));
        dtScores_list.push_back(std::vector<double>(D));

        // pointers
        auto& gtIg = gtIgnore_list.back();
        auto& dtIg = dtIgnore_list.back();
        auto& dtm = dtMatches_list.back();
        auto& dtScores = dtScores_list.back();
        auto ious = (ious_map[key(imgId, catId)].size() == 0) ? nullptr : ious_map[key(imgId,catId)].data();

        // set ignores
        for (int g = 0; g < G; g++) {
          gtIg[g] = (gtsm.ignore[g] || gtsm.area[g]<aRng0 || gtsm.area[g]>aRng1) ? 1 : 0;
        }
        // get sorting indices
        auto gtind = sort_indices(gtIg, std::less<double>());
        auto dtind = sort_indices(dtsm.score);

        // if not len(ious)==0:
        if(I != 0) {
          for (int t = 0; t < T; t++) {
            double thresh = iouThrs_ptr[t];
            for (int d = 0; d < D; d++) {
              double iou = thresh < (1-1e-10) ? thresh : (1-1e-10);
              int m = -1;
              for (int g = 0; g < G; g++) {
                // if this gt already matched, and not a crowd, continue
                if((gtm[t * G + g]>0) && (gtsm.iscrowd[gtind[g]]==0))
                  continue;
                // if dt matched to reg gt, and on ignore gt, stop
                if((m>-1) && (gtIg[gtind[m]]==0) && (gtIg[gtind[g]]==1))
                  break;
                // continue to next gt unless better match made
                double val = ious[d + I * gtind[g]];
                if(val < iou)
                  continue;
                // if match successful and best so far, store appropriately
                iou=val;
                m=g;
              }
              // if match made store id of match for both dt and gt
              if(m ==-1)
                continue;
              dtIg[t * D + d] = gtIg[gtind[m]];
              dtm[t * D + d]  = gtsm.id[gtind[m]];
              gtm[t * G + m]  = dtsm.id[dtind[d]];
            }
          }
        }
        // set unmatched detections outside of area range to ignore
        for (int d = 0; d < D; d++) {
          float val = dtsm.area[dtind[d]];
          double x3 = (val<aRng0 || val>aRng1);
          for (int t = 0; t < T; t++) {
            double x1 = dtIg[t * D + d];
            double x2 = dtm[t * D + d];
            double res = x1 || ((x2==0) && x3);
            dtIg[t * D + d] = res;
          }
        }
        // store results for given image and category
        for (int d = 0; d < D; d++) {
          dtScores[d] = dtsm.score[dtind[d]];
        }
      }
      // accumulate
      accumulate(iouThrs_ptr.size(), areaRngs.size(), maxDets, recThrs,
                 precision,
                 recall,
                 scores,
                 catids.size(), imgids.size(), recThrs.size(), maxDets.size(), c, a,
                 gtIgnore_list,
                 dtIgnore_list,
                 dtMatches_list,
                 dtScores_list);
    }
  }

  // clear arrays
  std::unordered_map<size_t, std::vector<double>>().swap(ious_map);
  //std::unordered_map<size_t, data_struct>().swap(gts_map);
  std::unordered_map<size_t, data_struct>().swap(dts_map);
  //std::unordered_map<size_t, image_struct>().swap(imgsdt);
  //std::unordered_map<size_t, anns_struct>().swap(annsdt);
  //std::unordered_map<size_t, std::vector<anns_struct>>().swap(dtimgToAnns);

  // dictionary
  py::dict dictret;
  py::list l;
  l.append(T);
  l.append(R);
  l.append(K);
  l.append(A);
  l.append(M);
  dictret["counts"] = l;
  dictret["precision"] = py::array_t<double>({T,R,K,A,M},{R*K*A*M*8,K*A*M*8,A*M*8,M*8,8},precision.data());
  dictret["recall"] = py::array_t<double>({T,K,A,M},{K*A*M*8,A*M*8,M*8,8},recall.data());
  dictret["scores"] = py::array_t<double>({T,R,K,A,M},{R*K*A*M*8,K*A*M*8,A*M*8,M*8,8},scores.data());

  py::array_t<int64_t> imgidsret = py::array_t<int64_t>({imgids.size()},{8}, imgids.data());
  py::array_t<int64_t> catidsret = py::array_t<int64_t>({catids.size()},{8}, catids.data());

  return std::tuple<py::array_t<int64_t>,py::array_t<int64_t>,py::dict>(imgidsret,catidsret,dictret);
}


template <typename T>
std::vector<T> assemble_array(const std::vector<std::vector<T>>& list, size_t nrows, size_t maxDet, const std::vector<size_t>& indices) {
  std::vector<T> q;
  // Need to get N_rows from an entry in order to compute output size
  // copy first maxDet entries from each entry -> array
  for (size_t e = 0; e < list.size(); ++e) {
    auto arr = list[e];
    size_t cols = arr.size() / nrows;
    size_t ncols = std::min(maxDet, cols);
    for (size_t j = 0; j < ncols; ++j) {
    for (size_t i = 0; i < nrows; ++i) {
        q.push_back(arr[i * cols + j]);
      }
    }
  }
  // now we've done that, copy the relevant entries based on indices
  std::vector<T> res(indices.size() * nrows);
  for (size_t i = 0; i < nrows; ++i) {
    for (size_t j = 0; j < indices.size(); ++j) {
      res[i * indices.size() + j] = q[indices[j] * nrows + i];
    }
  }

  return res;
}

void accumulate(int T, int A, const std::vector<int>& maxDets, const std::vector<double>& recThrs,
                std::vector<double>& precision,
                std::vector<double>& recall,
                std::vector<double>& scores,
                int K, int I, int R, int M, int k, int a,
                const std::vector<std::vector<int64_t>>& gtignore,
                const std::vector<std::vector<double>>& dtignore,
                const std::vector<std::vector<double>>& dtmatches,
                const std::vector<std::vector<double>>& dtscores) {
  if (dtscores.size() == 0) return;

  int npig = 0;
  for (size_t e = 0; e < gtignore.size(); ++e) {
    auto ignore = gtignore[e];
    for (size_t j = 0; j < ignore.size(); ++j) {
      if(ignore[j] == 0) npig++;
    }
  }
  if (npig == 0) return;

  for (int m = 0; m < M; ++m) {
    // Concatenate first maxDet scores in each evalImg entry, -ve and sort w/indices
    std::vector<double> dtScores;
    for (size_t e = 0; e < dtscores.size(); ++e) {
      auto score = dtscores[e];
      for (size_t j = 0; j < std::min(score.size(), static_cast<size_t>(maxDets[m])); ++j) {
        dtScores.push_back(score[j]);
      }
    }

    // get sorted indices of scores
    auto indices = stable_sort_indices(dtScores);
    std::vector<double> dtScoresSorted(dtScores.size());
    for (size_t j = 0; j < indices.size(); ++j) {
      dtScoresSorted[j] = dtScores[indices[j]];
    }

    auto dtm = assemble_array<double>(dtmatches, T, maxDets[m], indices);
    auto dtIg = assemble_array<double>(dtignore, T, maxDets[m], indices);

    int nrows = indices.size() ? dtm.size()/indices.size() : 0;
    std::vector<double> tp_sum(indices.size() * nrows);
    std::vector<double> fp_sum(indices.size() * nrows);
    for (int i = 0; i < nrows; ++i) {
      size_t tsum = 0, fsum = 0;
      for (size_t j = 0; j < indices.size(); ++j) {
        int index = i * indices.size() + j;
        tsum += (dtm[index]) && (!dtIg[index]);
        fsum += (!dtm[index]) && (!dtIg[index]);
        tp_sum[index] = tsum;
        fp_sum[index] = fsum;
      }
    }

    double eps = 2.220446049250313e-16; //std::numeric_limits<double>::epsilon();
    for (int t = 0; t < nrows; ++t) {
      // nd = len(tp)
      int nd = indices.size();
      std::vector<double> rc(indices.size());
      std::vector<double> pr(indices.size());
      for (size_t j = 0; j < indices.size(); ++j) {
        int index = t * indices.size() + j;
        // rc = tp / npig
        rc[j] = tp_sum[index] / npig;
        // pr = tp / (fp+tp+np.spacing(1))
        pr[j] = tp_sum[index] / (fp_sum[index]+tp_sum[index]+eps);
      }

      recall[t*K*A*M + k*A*M + a*M + m] = nd ? rc[indices.size()-1] : 0;

      for (int i = nd-1; i > 0; --i) {
        if (pr[i] > pr[i-1]) {
          pr[i-1] = pr[i];
        }
      }

      std::vector<int> inds(recThrs.size());
      for (size_t i = 0; i < recThrs.size(); i++) {
        auto it = std::lower_bound(rc.begin(), rc.end(), recThrs[i]);
        inds[i] = it - rc.begin();
      }

      for (size_t i = 0; i < inds.size(); i++) {
        size_t pi = inds[i];
        size_t index = t*R*K*A*M + i*K*A*M + k*A*M + a*M + m;
        if (pi < pr.size()) {
          precision[index] = pr[pi];
          scores[index] = dtScoresSorted[pi];
        }
      }
    }
  }
}


void bbIou(const double *dt, const double *gt, const int m, const int n, const int *iscrowd, double *o) {
  for(int g=0; g<n; g++ ) {
    const double* G = gt+g*4;
    const double ga = G[2]*G[3];
    const double crowd = iscrowd!=NULL && iscrowd[g];
    for(int d=0; d<m; d++ ) {
      const double* D = dt+d*4;
      const double da = D[2]*D[3];
      o[g*m+d]=0;
      double w = fmin(D[2]+D[0],G[2]+G[0])-fmax(D[0],G[0]);
      if(w <= 0)
        continue;
      double h = fmin(D[3]+D[1],G[3]+G[1])-fmax(D[1],G[1]);
      if(h <= 0)
        continue;
      double i = w*h;
      double u = crowd ? da : da+ga-i;
      o[g*m+d] = i/u;
    }
  }
}

void rleInit( RLE *R, unsigned long h, unsigned long w, unsigned long m, unsigned int *cnts ) {
  R->h=h;
  R->w=w;
  R->m=m;
  R->cnts = (m==0) ? 0 : (unsigned int*)malloc(sizeof(unsigned int)*m);
  if(cnts) {
    for(unsigned long j=0; j<m; j++){
      R->cnts[j]=cnts[j];
    }
  }
}

void rleFree( RLE *R ) {
  free(R->cnts);
  R->cnts=0;
}


void rleFrString( RLE *R, char *s, unsigned long h, unsigned long w ) {
  unsigned long m=0, p=0, k;
  long x;
  int more;
  unsigned int *cnts;
  while( s[m] ){
    m++;
  }
  cnts = (unsigned int*)malloc(sizeof(unsigned int)*m);
  m = 0;
  while( s[p] ) {
    x=0; k=0; more=1;
    while( more ) {
      char c=s[p]-48; x |= (c & 0x1f) << 5*k;
      more = c & 0x20; p++; k++;
      if(!more && (c & 0x10)) x |= -1 << 5*k;
    }
    if(m>2) {
      x += static_cast<long>(cnts[m-2]);
    }
    cnts[m++] = static_cast<unsigned int>(x);
  }
  rleInit(R, h, w, m, cnts);
  free(cnts);
}


unsigned int umin( unsigned int a, unsigned int b ) { return (a<b) ? a : b; }
unsigned int umax( unsigned int a, unsigned int b ) { return (a>b) ? a : b; }

void rleArea( const RLE *R, unsigned long n, unsigned int *a ) {
  for(unsigned long i=0; i<n; i++ ) {
    a[i]=0;
    for(unsigned long j=1; j<R[i].m; j+=2 ) {
      a[i]+=R[i].cnts[j];
    }
  }
}

void rleToBbox( const RLE *R, double* bb, unsigned long n ) {
  for(unsigned long i=0; i<n; i++ ) {
    unsigned int h, w, x, y, xs, ys, xe=0, ye=0, xp=0, cc=0, t;
    unsigned long j, m;
    h = static_cast<unsigned int>(R[i].h);
    w = static_cast<unsigned int>(R[i].w);
    m = (static_cast<unsigned long>(R[i].m/2))*2;
    xs=w;
    ys=h;
    if(m==0) {
      bb[4*i+0]=bb[4*i+1]=bb[4*i+2]=bb[4*i+3]=0;
      continue;
    }
    for( j=0; j<m; j++ ) {
      cc += R[i].cnts[j];
      t = cc-j%2;
      y = t%h;
      x = (t-y)/h;
      if(j%2==0) {
        xp = x;
      } else if(xp<x) {
        ys = 0;
        ye = h-1;
      }
      xs = umin(xs, x);
      xe = umax(xe, x);
      ys = umin(ys, y);
      ye = umax(ye, y);
    }
    bb[4*i+0] = xs;
    bb[4*i+1] = ys;
    bb[4*i+2] = xe-xs+1;
    bb[4*i+3] = ye-ys+1;
  }
}

void rleIou(const RLE *dt, const RLE *gt, const int m, const int n, const int *iscrowd, double *o ) {
  // TODO(ahmadki): smart pointers
  double *db=(double*)malloc(sizeof(double)*m*4);
  double *gb=(double*)malloc(sizeof(double)*n*4);
  rleToBbox(dt, db, m);
  rleToBbox(gt, gb, n);
  bbIou(db, gb, m, n, iscrowd, o);
  free(db);
  free(gb);
  for(int g=0; g<n; g++ ) {
    for(int d=0; d<m; d++ ) {
      if(o[g*m+d]>0) {
        int crowd = iscrowd!=NULL && iscrowd[g];
        if(dt[d].h!=gt[g].h || dt[d].w!=gt[g].w) {
          o[g*m+d]=-1;
          continue;
        }
        unsigned long ka, kb, a, b; uint c, ca, cb, ct, i, u; int va, vb;
        ca=dt[d].cnts[0]; ka=dt[d].m; va=vb=0;
        cb=gt[g].cnts[0]; kb=gt[g].m; a=b=1; i=u=0; ct=1;
        while(ct > 0) {
          c=umin(ca,cb);
          if(va||vb) {
            u+=c;
            if(va&&vb) {
              i+=c;
            }
          }
          ct = 0;
          ca -=c;
          if(!ca && a<ka) {
            ca=dt[d].cnts[a++]; va=!va;
          }
          ct+=ca;
          cb -=c;
          if(!cb && b<kb) {
            cb=gt[g].cnts[b++];
            vb=!vb;
          }
          ct += cb;
        }
        if(i==0) {
          u=1;
        } else if(crowd) {
          rleArea(dt+d, 1, &u);
        }
        o[g*m+d] = static_cast<double>(i)/static_cast<double>(u);
      }
    }
  }
}

void compute_iou(std::string iouType, int maxDet, int useCats) {
  assert(iouType=="bbox"||iouType=="segm");
  assert(useCats > 0);

  // TODO(ahmadki): parallelize
  // #pragma omp parallel for num_threads(nthreads)
  for(size_t i = 0; i < imgids.size(); i++) {
    for(size_t c = 0; c < catids.size(); c++) {
      const int catId = catids[c];
      const int imgId = imgids[i];
      const auto gtsm = gts_map[key(imgId, catId)];
      const auto dtsm = dts_map[key(imgId, catId)];
      const auto G = gtsm.id.size();
      const auto D = dtsm.id.size();
      const int m = std::min(D, static_cast<size_t>(maxDet));
      const int n = G;

      if(m==0 || n==0) {
        ious_map[key(imgId,catId)] = std::vector<double>();
        // ious_map[key(imgId, catId)] = std::make_shared<std::vector<double>>();
        continue;
      }

      auto inds = sort_indices(dtsm.score);

      if (iouType == "bbox") {
        std::vector<double> d;
        for (size_t i = 0; i < m; i++) {
          auto arr = dtsm.bbox[inds[i]];
          for (size_t j = 0; j < arr.size(); j++) {
            d.push_back(static_cast<double>(arr[j]));
          }
        }

        ious_map[key(imgId,catId)] = std::vector<double>(m*n);
        // ious_map[key(imgId, catId)] = std::make_shared<std::vector<double>>(m*n);
        bbIou(d.data(), gtbboxes[i*catids.size()+c].data(), m, n, gtsm.iscrowd.data(), ious_map[key(imgId, catId)].data());
      } else {
        std::vector<RLE> d(m);
        for (size_t i = 0; i < m; i++) {
          auto size = dtsm.segm_size[i];
          auto str = dtsm.segm_counts[inds[i]];
          char *val = new char[str.length() + 1];
          strcpy(val, str.c_str());
          rleFrString(&d[i],val,size[0],size[1]);
          delete [] val;
        }

        ious_map[key(imgId,catId)] = std::vector<double>(m*n);
        // ious_map[key(imgId, catId)] = std::make_shared<std::vector<double>>(m*n);
        rleIou(d.data(), gtsegm[i*catids.size()+c].data(), m, n, gtsm.iscrowd.data(), ious_map[key(imgId, catId)].data());
        for (size_t i = 0; i < d.size(); i++) {free(d[i].cnts);}
      }
    }
  }
}

std::string rleToString( const RLE *R ) {
  /* Similar to LEB128 but using 6 bits/char and ascii chars 48-111. */
  unsigned long i, m=R->m, p=0; long x; int more;
  char *s=(char*)malloc(sizeof(char)*m*6);
  for( i=0; i<m; i++ ) {
    x=(long) R->cnts[i]; if(i>2) x-=(long) R->cnts[i-2]; more=1;
    while( more ) {
      char c=x & 0x1f; x >>= 5; more=(c & 0x10) ? x!=-1 : x!=0;
      if(more) c |= 0x20; c+=48; s[p++]=c;
    }
  }
  s[p]=0;
  std::string str = std::string(s);
  free(s);
  return str;
}

std::string frUncompressedRLE(std::vector<int> cnts, std::vector<int> size, int h, int w) {
  unsigned int *data = (unsigned int*) malloc(cnts.size() * sizeof(unsigned int));
  for(size_t i = 0; i < cnts.size(); i++) {
    data[i] = static_cast<unsigned int>(cnts[i]);
  }
  RLE R;// = RLE(size[0],size[1],cnts.size(),data);
  R.h = size[0];
  R.w = size[1];
  R.m = cnts.size();
  R.cnts = data;
  std::string str = rleToString(&R);
  free(data);
  return str;
}

int uintCompare(const void *a, const void *b) {
  unsigned int c=*((unsigned int*)a), d=*((unsigned int*)b); return c>d?1:c<d?-1:0;
}

void rleFrPoly(RLE *R, const double *xy, int k, int h, int w ) {
  /* upsample and get discrete points densely along entire boundary */
  int j, m=0;
  double scale=5;
  unsigned int *a, *b;
  int *x = (int*)malloc(sizeof(int)*(k+1));
  int *y = (int*)malloc(sizeof(int)*(k+1));
  for(j=0; j<k; j++) x[j] = static_cast<int>(scale*xy[j*2+0]+.5); x[k] = x[0];
  for(j=0; j<k; j++) y[j] = static_cast<int>(scale*xy[j*2+1]+.5); y[k] = y[0];
  for(j=0; j<k; j++) m += umax(abs(x[j]-x[j+1]),abs(y[j]-y[j+1]))+1;
  int *u=(int*)malloc(sizeof(int)*m);
  int *v=(int*)malloc(sizeof(int)*m);
  m = 0;
  for( j=0; j<k; j++ ) {
    int xs=x[j], xe=x[j+1], ys=y[j], ye=y[j+1], dx, dy, t, d;
    int flip; double s; dx=abs(xe-xs); dy=abs(ys-ye);
    flip = (dx>=dy && xs>xe) || (dx<dy && ys>ye);
    if(flip) { t=xs; xs=xe; xe=t; t=ys; ys=ye; ye=t; }
    s = dx>=dy ? static_cast<double>(ye-ys)/dx : static_cast<double>(xe-xs)/dy;
    if(dx>=dy) for( d=0; d<=dx; d++ ) {
      t=flip?dx-d:d; u[m]=t+xs; v[m]=static_cast<int>(ys+s*t+.5); m++;
    } else for( d=0; d<=dy; d++ ) {
      t=flip?dy-d:d; v[m]=t+ys; u[m]=static_cast<int>(xs+s*t+.5); m++;
    }
  }
  /* get points along y-boundary and downsample */
  free(x); free(y); k=m; m=0; double xd, yd;
  x=(int*)malloc(sizeof(int)*k); y=(int*)malloc(sizeof(int)*k);
  for( j=1; j<k; j++ ) if(u[j]!=u[j-1]) {
    xd=static_cast<double>(u[j]<u[j-1]?u[j]:u[j]-1); xd=(xd+.5)/scale-.5;
    if( floor(xd)!=xd || xd<0 || xd>w-1 ) continue;
    yd=static_cast<double>(v[j]<v[j-1]?v[j]:v[j-1]); yd=(yd+.5)/scale-.5;
    if(yd<0) yd=0; else if(yd>h) yd=h; yd=ceil(yd);
    x[m]=static_cast<int>(xd); y[m]=static_cast<int>(yd); m++;
  }
  /* compute rle encoding given y-boundary points */
  k=m; a=(unsigned int*)malloc(sizeof(unsigned int)*(k+1));
  for( j=0; j<k; j++ ) a[j]=static_cast<unsigned int>(x[j]*static_cast<int>(h)+y[j]);
  a[k++]=static_cast<unsigned int>(h*w); free(u); free(v); free(x); free(y);
  qsort(a,k,sizeof(unsigned int),uintCompare); unsigned int p=0;
  for( j=0; j<k; j++ ) { unsigned int t=a[j]; a[j]-=p; p=t; }
  b=(unsigned int*)malloc(sizeof(unsigned int)*k); j=m=0; b[m++]=a[j++];
  while(j<k) if(a[j]>0) b[m++]=a[j++]; else {
    j++; if(j<k) b[m-1]+=a[j++]; }
  rleInit(R,h,w,m,b); free(a); free(b);
}

void rleMerge( const RLE *R, RLE *M, unsigned long n, int intersect ) {
  unsigned int *cnts, c, ca, cb, cc, ct; int v, va, vb, vp;
  unsigned long i, a, b, h=R[0].h, w=R[0].w, m=R[0].m; RLE A, B;
  if(n==0) { rleInit(M,0,0,0,0); return; }
  if(n==1) { rleInit(M,h,w,m,R[0].cnts); return; }
  cnts = (unsigned int*)malloc(sizeof(unsigned int)*(h*w+1));
  for( a=0; a<m; a++ ) cnts[a]=R[0].cnts[a];
  for( i=1; i<n; i++ ) {
    B=R[i]; if(B.h!=h||B.w!=w) { h=w=m=0; break; }
    rleInit(&A,h,w,m,cnts); ca=A.cnts[0]; cb=B.cnts[0];
    v=va=vb=0; m=0; a=b=1; cc=0; ct=1;
    while( ct>0 ) {
      c=umin(ca,cb); cc+=c; ct=0;
      ca-=c; if(!ca && a<A.m) { ca=A.cnts[a++]; va=!va; } ct+=ca;
      cb-=c; if(!cb && b<B.m) { cb=B.cnts[b++]; vb=!vb; } ct+=cb;
      vp=v; if(intersect) v=va&&vb; else v=va||vb;
      if( v!=vp||ct==0 ) { cnts[m++]=cc; cc=0; }
    }
    rleFree(&A);
  }
  rleInit(M,h,w,m,cnts); free(cnts);
}

void rlesInit( RLE **R, unsigned long n ) {
  unsigned long i; *R = (RLE*) malloc(sizeof(RLE)*n);
  for(i=0; i<n; i++) rleInit((*R)+i,0,0,0,0);
}

std::string frPoly(std::vector<std::vector<double>> poly, int h, int w) {
  size_t n = poly.size();
  RLE *Rs;
  rlesInit(&Rs,n);
  for (size_t i = 0; i < n; i++) {
    double* p = (double*)malloc(sizeof(double)*poly[i].size());
    for (size_t j = 0; j < poly[i].size(); j++) {
      p[j] = static_cast<double>(poly[i][j]);
    }
    rleFrPoly(&Rs[i],p,int(poly[i].size()/2),h,w);
    free(p);
  }
  // _toString
  /*std::vector<char*> string;
  for (size_t i = 0; i < n; i++) {
    char* c_string = rleToString(&Rs[i]);
    string.push_back(c_string);
  }
  // _frString
  RLE *Gs;
  rlesInit(&Gs,n);
  for (size_t i = 0; i < n; i++) {
    rleFrString(&Gs[i],string[i],h,w);
  }*/
  // merge(rleObjs, intersect=0)
  RLE R;
  int intersect = 0;
  rleMerge(Rs, &R, n, intersect);
  std::string str = rleToString(&R);
  for (size_t i = 0; i < n; i++) {free(Rs[i].cnts);}
  free(Rs);
  return str;
}

unsigned int
area(std::vector<int>& size, std::string& counts) {
  // _frString
  RLE *Rs;
  rlesInit(&Rs,1);
  char *str = new char[counts.length() + 1];
  strcpy(str, counts.c_str());
  rleFrString(&Rs[0],str,size[0],size[1]);
  delete [] str;
  unsigned int a;
  rleArea(Rs, 1, &a);
  for (size_t i = 0; i < 1; i++) {free(Rs[i].cnts);}
  free(Rs);
  return a;
}

std::vector<float>
toBbox(std::vector<int>& size, std::string& counts) {
  // _frString
  RLE *Rs;
  rlesInit(&Rs,1);
  char *str = new char[counts.length() + 1];
  strcpy(str, counts.c_str());
  rleFrString(&Rs[0],str,size[0],size[1]);
  delete [] str;

  std::vector<double> bb(4*1);
  rleToBbox(Rs, bb.data(), 1);
  std::vector<float> bbf(bb.size());
  for (size_t i = 0; i < bb.size(); i++) {
    bbf[i] = static_cast<float>(bb[i]);
  }
  for (size_t i = 0; i < 1; i++) {free(Rs[i].cnts);}
  free(Rs);
  return bbf;
}

void annToRLE(anns_struct& ann, std::vector<std::vector<int>> &size, std::vector<std::string> &counts, int h, int w) {
  auto is_segm_list = ann.segm_list.size()>0;
  auto is_cnts_list = is_segm_list ? 0 : ann.segm_counts_list.size()>0;

  if (is_segm_list) {
    std::vector<int> segm_size{h,w};
    auto cnts = ann.segm_list;
    auto segm_counts = frPoly(cnts, h, w);
    size.push_back(segm_size);
    counts.push_back(segm_counts);
  } else if (is_cnts_list) {
    auto segm_size = ann.segm_size;
    auto cnts = ann.segm_counts_list;
    auto segm_counts = frUncompressedRLE(cnts, segm_size, h, w);
    size.push_back(segm_size);
    counts.push_back(segm_counts);
  } else {
    auto segm_size = ann.segm_size;
    auto segm_counts = ann.segm_counts_str;
    size.push_back(segm_size);
    counts.push_back(segm_counts);
  }
}

void getAnnsIds(std::unordered_map<size_t, std::vector<anns_struct>>& imgToAnns, std::unordered_map<size_t, anns_struct>& anns,
                      std::vector<int64_t>& ids, std::vector<int64_t>& imgIds, std::vector<int64_t>& catIds) {
  for (size_t i = 0; i < imgIds.size(); i++) {
    auto hasimg = imgToAnns.find(imgIds[i]) != imgToAnns.end();
    if (hasimg) {
      auto anns = imgToAnns[imgIds[i]];
      for (size_t j = 0; j < anns.size(); j++) {
 //       auto catid = anns[j].category_id;
//        auto hascat = (std::find(catIds.begin(), catIds.end(), catid) != catIds.end());  // set might be faster? does it matter?
//        if (hascat) {
          //auto area = py::cast<float>(anns[j]["area"]);
          // some indices can have float values, so cast to double first
          ids.push_back(anns[j].id);
//        }
      }
    }
  }
}

void cpp_load_res_numpy(py::dict dataset, std::vector<std::vector<float>> data) {
/*void cpp_load_res_numpy(py::dict dataset, py::array data) {
  auto buf = data.request();
  //float* data_ptr = (float*)buf.ptr;
  double* data_ptr = (double*)buf.ptr;//sometimes predictions are in double?
  size_t size = buf.shape[0];*/
  for (size_t i = 0; i < data.size(); i++) {
/*  for (size_t i = 0; i < size; i++) {
    auto datai = &data_ptr[i*7];*/
    anns_struct ann;
    ann.image_id = int(data[i][0]);
    //ann.image_id = int(datai[0]);
    ann.bbox = std::vector<float>{data[i][1], data[i][2], data[i][3], data[i][4]};
    //ann.bbox = std::vector<float>{(float)datai[1], (float)datai[2], (float)datai[3], (float)datai[4]};
    ann.score = data[i][5];
    //ann.score = datai[5];
    ann.category_id = data[i][6];
    //ann.category_id = datai[6];
    auto bb = ann.bbox;
    auto x1 = bb[0];
    auto x2 = bb[0]+bb[2];
    auto y1 = bb[1];
    auto y2 = bb[1]+bb[3];
    ann.segm_list = std::vector<std::vector<double>>{{x1, y1, x1, y2, x2, y2, x2, y1}};
    ann.area = bb[2]*bb[3];
    ann.id = i+1;
    ann.iscrowd = 0;

    auto k = key(ann.image_id,ann.category_id);
    data_struct* tmp = &dts_map[k];
    tmp->area.push_back(ann.area);
    tmp->iscrowd.push_back(ann.iscrowd);
    tmp->bbox.push_back(ann.bbox);
    tmp->score.push_back(ann.score);
    tmp->id.push_back(ann.id);
  }
}

void cpp_load_res(py::dict dataset, std::vector<py::dict> anns) {
  auto iscaption = anns[0].contains("caption");
  auto isbbox = anns[0].contains("bbox") && (py::cast<std::vector<float>>(anns[0]["bbox"]).size() > 0);
  auto issegm = anns[0].contains("segmentation");
  assert(!iscaption && (isbbox||issegm));

  if(isbbox) {
    for(size_t i = 0; i < anns.size(); i++) {
      anns_struct ann;
      ann.image_id = py::cast<int>(anns[i]["image_id"]);
      ann.category_id = py::cast<int64_t>(anns[i]["category_id"]);
      auto bb = py::cast<std::vector<float>>(anns[i]["bbox"]);
      ann.bbox = bb;
      auto x1 = bb[0];
      auto x2 = bb[0]+bb[2];
      auto y1 = bb[1];
      auto y2 = bb[1]+bb[3];
      if (!issegm) {
        ann.segm_list = std::vector<std::vector<double>>{{x1, y1, x1, y2, x2, y2, x2, y1}};
      } else { // do we need all of these?
        auto is_segm_list = py::isinstance<py::list>(anns[i]["segmentation"]);
        auto is_cnts_list = is_segm_list ? 0 : py::isinstance<py::list>(anns[i]["segmentation"]["counts"]);
        if (is_segm_list) {
          ann.segm_list = py::cast<std::vector<std::vector<double>>>(anns[i]["segmentation"]);
        } else if (is_cnts_list) {
          ann.segm_size = py::cast<std::vector<int>>(anns[i]["segmentation"]["size"]);
          ann.segm_counts_list = py::cast<std::vector<int>>(anns[i]["segmentation"]["counts"]);
        } else {
          ann.segm_size = py::cast<std::vector<int>>(anns[i]["segmentation"]["size"]);
          ann.segm_counts_str = py::cast<std::string>(anns[i]["segmentation"]["counts"]);
        }
      }
      ann.score = py::cast<float>(anns[i]["score"]);
      ann.area = bb[2]*bb[3];
      ann.id = i+1;
      ann.iscrowd = 0;
      //annsdt[ann.id] = ann;
      //dtimgToAnns[(size_t)ann.image_id].push_back(ann);
      auto k = key(ann.image_id,ann.category_id);
      data_struct* tmp = &dts_map[k];
      tmp->area.push_back(ann.area);
      tmp->iscrowd.push_back(ann.iscrowd);
      tmp->bbox.push_back(ann.bbox);
      tmp->score.push_back(ann.score);
      tmp->id.push_back(ann.id);
    }
  } else {
    std::unordered_map<size_t, image_struct> imgsdt;
    auto imgs = py::cast<std::vector<py::dict>>(dataset["images"]);
    for (size_t i = 0; i < imgs.size(); i++) {
      image_struct img;
      img.id = (size_t)py::cast<double>(imgs[i]["id"]);
      img.h = py::cast<int>(imgs[i]["height"]);
      img.w = py::cast<int>(imgs[i]["width"]);
      imgsdt[img.id] = img;
    }
    for (size_t i = 0; i < anns.size(); i++) {
      anns_struct ann;
      ann.image_id = py::cast<int>(anns[i]["image_id"]);
      ann.category_id = py::cast<int64_t>(anns[i]["category_id"]);
      // now only support compressed RLE format as segmentation results
      ann.segm_size = py::cast<std::vector<int>>(anns[i]["segmentation"]["size"]);
      ann.segm_counts_str = py::cast<std::string>(anns[i]["segmentation"]["counts"]);
      ann.area = area(ann.segm_size,ann.segm_counts_str);
      if(!anns[0].contains("bbox")) {
        ann.bbox = toBbox(ann.segm_size,ann.segm_counts_str);
      }
      ann.score = py::cast<float>(anns[i]["score"]);
      ann.id = i+1;
      ann.iscrowd = 0;
      //annsdt[ann.id] = ann;
      //dtimgToAnns[(size_t)ann.image_id].push_back(ann);
      auto k = key(ann.image_id,ann.category_id);
      data_struct* tmp = &dts_map[k];
      tmp->area.push_back(ann.area);
      tmp->iscrowd.push_back(ann.iscrowd);
      tmp->bbox.push_back(ann.bbox);
      tmp->score.push_back(ann.score);
      tmp->id.push_back(ann.id);
      // convert ground truth to mask if iouType == 'segm'
      auto h = imgsdt[static_cast<size_t>(ann.image_id)].h;
      auto w = imgsdt[static_cast<size_t>(ann.image_id)].w;
      annToRLE(ann,tmp->segm_size,tmp->segm_counts,h,w);
    }
  }
}

void cpp_create_index(py::dict dataset) {
  if (imgsgt.size()>0 && imgids.size()>0 && catids.size()>0) {
    printf("GT annotations already exist!\n");
    return;
    // clear arrays
    /*printf("GT annotations already exist, cleanup and create again...\n");
    std::unordered_map<size_t, data_struct>().swap(gts_map);
    std::unordered_map<size_t, image_struct>().swap(imgsgt);
    std::vector<int64_t>().swap(imgids);
    std::vector<int64_t>().swap(catids);*/
  }

  auto imgs = py::cast<std::vector<py::dict>>(dataset["images"]);
  for (size_t i = 0; i < imgs.size(); i++) {
    image_struct img;
    img.id = (size_t)py::cast<double>(imgs[i]["id"]);
    img.h = py::cast<int>(imgs[i]["height"]);
    img.w = py::cast<int>(imgs[i]["width"]);
    imgsgt[img.id] = img;
    imgids.push_back(img.id);
  }
  auto cats = py::cast<std::vector<py::dict>>(dataset["categories"]);
  for (size_t i = 0; i < cats.size(); i++) {
    auto catid = py::cast<int>(cats[i]["id"]);
    catids.push_back(catid);
  }

  auto issegm = dataset.contains("segmentation");
  auto anns = py::cast<std::vector<py::dict>>(dataset["annotations"]);
  for (size_t i = 0; i < anns.size(); i++) {
    anns_struct ann;
    ann.image_id = py::cast<int>(anns[i]["image_id"]);
    ann.category_id = py::cast<int64_t>(anns[i]["category_id"]);
    ann.id = (size_t)py::cast<double>(anns[i]["id"]);
    ann.area = py::cast<float>(anns[i]["area"]);
    ann.iscrowd = py::cast<int>(anns[i]["iscrowd"]);
    /*auto has_score = (anns[i].contains("score"));
    if (has_score) {
      ann.score = py::cast<float>(anns[i]["score"]);
    }*/
    ann.bbox = py::cast<std::vector<float>>(anns[i]["bbox"]);

    if (issegm) {
        auto is_segm_list = py::isinstance<py::list>(anns[i]["segmentation"]);
        auto is_cnts_list = is_segm_list ? 0 : py::isinstance<py::list>(anns[i]["segmentation"]["counts"]);

        if (is_segm_list) {
            ann.segm_list = py::cast<std::vector<std::vector<double>>>(anns[i]["segmentation"]);
        } else if (is_cnts_list) {
            ann.segm_size = py::cast<std::vector<int>>(anns[i]["segmentation"]["size"]);
            ann.segm_counts_list = py::cast<std::vector<int>>(anns[i]["segmentation"]["counts"]);
        } else {
            ann.segm_size = py::cast<std::vector<int>>(anns[i]["segmentation"]["size"]);
            ann.segm_counts_str = py::cast<std::string>(anns[i]["segmentation"]["counts"]);
        }
    }

    auto k = key(ann.image_id, ann.category_id);
    data_struct* tmp = &gts_map[k];
    tmp->area.push_back(ann.area);
    tmp->iscrowd.push_back(ann.iscrowd);
    tmp->bbox.push_back(ann.bbox);
    tmp->ignore.push_back(ann.iscrowd!=0);
    //tmp->score.push_back(ann.score);
    tmp->id.push_back(ann.id);
    auto h = imgsgt[static_cast<size_t>(ann.image_id)].h;
    auto w = imgsgt[static_cast<size_t>(ann.image_id)].w;
    annToRLE(ann, tmp->segm_size, tmp->segm_counts, h, w);
  }

  auto num_cats = cats.size();
  auto num_imgs = imgs.size();
  gtbboxes.resize(num_cats * num_imgs);
  gtsegm.resize(num_cats * num_imgs);

  for(size_t i = 0; i < num_imgs; i++) {
    for(size_t c = 0; c < num_cats; c++) {
      const int imgId = imgids[i];
      const int catId = catids[c];
      const auto gtsm = gts_map[key(imgId, catId)];
      const auto G = gtsm.id.size();
      if(G==0)
        continue;

      gtsegm[i*num_cats+c].resize(G);
      for (size_t g = 0; g < G; g++) {
        if (gtsm.segm_size[g].size()>0) {
          auto size = gtsm.segm_size[g];
          auto str = gtsm.segm_counts[g];
          // TODO(ahmadki): smart pointers
          char *val = new char[str.length() + 1];
          strcpy(val, str.c_str());
          rleFrString(&(gtsegm[i*num_cats+c][g]), val, size[0], size[1]);
          delete[] val;
        }

        for (size_t j = 0; j < gtsm.bbox[g].size(); j++)
          gtbboxes[i*num_cats+c].push_back(static_cast<double>(gtsm.bbox[g][j]));
      }
    }
  }

}


PYBIND11_MODULE(ext, m) {
  m.doc() = "pybind11 pycocotools plugin";
  m.def("cpp_evaluate", &cpp_evaluate, "");
  m.def("cpp_create_index", &cpp_create_index, "");
  m.def("cpp_load_res", &cpp_load_res, "");
  m.def("cpp_load_res_numpy", &cpp_load_res_numpy, "");
}
