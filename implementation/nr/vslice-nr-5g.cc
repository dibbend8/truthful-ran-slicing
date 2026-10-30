// vslice-nr-5g.cc  --  ns-3.46 + 5G-LENA (nr-4.1.1) closed-loop RAN-slicing scenario for
//                      the vslice verified-learning mechanism (ICC 2027).
//
// Derived from ICC/Overleaf/bridge/ns3/trustgate-nr-5g.cc (socket client, NDJSON loop,
// 3 slices x dedicated 5QI bearers x one BWP per slice, UMi channel, PF scheduler,
// eMBB mobility).  It changes the three things the ns-3 audit
// (ICC/docs/research/ns3_audit.md) found necessary:
//
//  (1) THE PRB BUDGET IS ENFORCED INSIDE THE NR MAC.  One gNB, three slices, one bandwidth
//      part (BWP) per slice.  Every BWP has the SAME numerology and the SAME width W, each
//      in its own adjacent operation band, so any single slice can be given the whole cell.
//      The xApp's per-slice quota n_s (PRBs) is applied at every epoch start with the public
//      API NrMacSchedulerNs3::SetDlNotchedRbgMask on the slice's BWP: only the first
//      round(n_s / RbPerRbg) RBGs are schedulable (new transmissions AND HARQ retx, see
//      nr-mac-scheduler-ofdma.cc:358 and nr-mac-scheduler-harq-rr.cc:251-262).  With
//      PowerAllocationType = UniformPowerAllocBw the per-RB PSD is P / N_RB whatever the
//      quota, so sum_s n_s <= N_RB radiates at most P: physically one shared carrier of
//      N_RB PRBs.  Default: W = 20 MHz, numerology 0 -> N_RB = 106 (== vslice Cell.B).
//      --enforce=shaper reproduces the old trustgate method (per-slice backhaul rate shaper
//      prb*capBits/slot, BWP masks frozen at an equal static split) for comparison.
//
//  (2) THE MEASUREMENT IS THE gNB's PER-SLICE PDCP SDU VOLUME.  Bytes of the inner IP
//      packets that the gNB receives on N3/S1-U (NrEpcGnbApplication "RxFromS1u", fired
//      immediately before the SDU enters PDCP), classified per slice by destination UE.
//      This is the simulator analogue of TS 28.552 DRB.PdcpSduVolumeDL per S-NSSAI.  With
//      --enforce=mask nothing between the source and PDCP depends on the allocation (open-
//      loop UDP, 10 Gb/s core), so it equals the replayed schedule byte-for-byte.  With
//      --enforce=shaper it does NOT (the shaper sits upstream of the gNB).
//
//  (3) DEMAND IS REPLAYED FROM A TRACE-DERIVED BYTE SCHEDULE.  --scheduleFile is a CSV with
//      one row per bin of --binMs ms (default 1 ms):
//          bin,<NS columns: IP bytes of slice s in the bin>                (aggregate), or
//          bin,<NS*U columns: IP bytes of slice s, UE u, slice-major>      (per UE)
//      A '#' line or a non-numeric first line is skipped.  Each bin's bytes are cut into IP
//      packets of at most maxPkt_s bytes (remainder < 40 B is carried to the next bin, so
//      the long-run volume is exact), spread uniformly inside the bin, and (aggregate mode)
//      dealt round-robin to the slice's UEs.  Each packet is UDP with a SeqTsHeader; IP size
//      == the scheduled bytes (payload = size - 20 IP - 8 UDP - 12 SeqTs).  The source of bin
//      b fires at  t0 - coreDelay + b*binMs,  so bin b reaches the gNB at t0 + b*binMs: the
//      PDCP counter window of epoch k is exactly the schedule window of epoch k.
//      The schedule is a function of the trace and the seed only -- never of the allocation
//      or of any report.
//
// Slice order is vslice's: 0 eMBB (5QI 8, PDB 300 ms), 1 URLLC (5QI 83, PDB 10 ms),
// 2 mMTC (5QI 9, PDB 300 ms).
//
// Wire protocol: NDJSON over TCP, ns-3 is the client (port 0 => offline, see below).
//   ns3    -> {"type":"hello","role":"ns3","scenario":"vslice-nr-5g","slices":3,"rb_total":N,
//              "rb_per_rbg":g,"enforce":"mask","measurement":"pdcp_sdu_bytes",...}
//   server -> {"type":"config"}                         (content ignored)
//   loop k = 0..epochs-1:
//     server -> {"type":"alloc","epoch":k,"prb":[n0,n1,n2],"cap_bits":[..] (shaper only)}
//     ns3    -> {"type":"kpi","epoch":k, per-slice arrays, see EndEpoch()}
//   server -> {"type":"done"}
// Offline mode (--port=0 --budgetFile=f.csv): row k of f.csv ("n0,n1,n2") is epoch k's quota
// (last row repeats).  --kpiFile=out.csv writes the per-epoch per-slice KPIs in both modes.
//
// Build:  cp vslice-nr-5g.cc ~/ns-3-dev/scratch/ && cd ~/ns-3-dev && ./ns3 build vslice-nr-5g

#include "ns3/antenna-module.h"
#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/internet-module.h"
#include "ns3/mobility-module.h"
#include "ns3/nr-module.h"
#include "ns3/point-to-point-module.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <map>
#include <sstream>
#include <string>
#include <vector>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("VsliceNr5g");

// --------------------------------------------------------------------------- //
// Minimal blocking NDJSON socket client (as in trustgate-nr-5g.cc, with full sends).
// --------------------------------------------------------------------------- //
class NdjsonSock
{
  public:
    bool Connect(const std::string& host, uint16_t port)
    {
        m_fd = socket(AF_INET, SOCK_STREAM, 0);
        if (m_fd < 0)
            return false;
        sockaddr_in a{};
        a.sin_family = AF_INET;
        a.sin_port = htons(port);
        inet_pton(AF_INET, host.c_str(), &a.sin_addr);
        return ::connect(m_fd, (sockaddr*)&a, sizeof(a)) == 0;
    }

    void SendLine(const std::string& s)
    {
        if (m_fd < 0)
            return;
        std::string m = s + "\n";
        size_t off = 0;
        while (off < m.size())
        {
            ssize_t n = ::send(m_fd, m.data() + off, m.size() - off, 0);
            if (n <= 0)
                return;
            off += size_t(n);
        }
    }

    std::string RecvLine()
    {
        std::string line;
        char c;
        while (m_fd >= 0 && ::recv(m_fd, &c, 1, 0) == 1)
        {
            if (c == '\n')
                return line;
            line.push_back(c);
        }
        return line;
    }

    void Close()
    {
        if (m_fd >= 0)
            ::close(m_fd);
        m_fd = -1;
    }

  private:
    int m_fd{-1};
};

static std::vector<double>
JsonArray(const std::string& js, const std::string& key)
{
    std::vector<double> out;
    auto p = js.find("\"" + key + "\"");
    if (p == std::string::npos)
        return out;
    p = js.find('[', p);
    auto e = js.find(']', p);
    if (p == std::string::npos || e == std::string::npos)
        return out;
    std::stringstream ss(js.substr(p + 1, e - p - 1));
    std::string tok;
    while (std::getline(ss, tok, ','))
        if (!tok.empty())
            out.push_back(std::stod(tok));
    return out;
}

static std::vector<double>
ParseList(const std::string& s)
{
    std::vector<double> v;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ','))
        if (!tok.empty())
            v.push_back(std::stod(tok));
    return v;
}

static std::vector<std::vector<double>>
ReadCsv(const std::string& path)
{
    std::vector<std::vector<double>> rows;
    std::ifstream f(path);
    NS_ABORT_MSG_IF(!f.good(), "cannot open " << path);
    std::string line;
    bool first = true;
    while (std::getline(f, line))
    {
        if (line.empty() || line[0] == '#')
            continue;
        if (first && !(std::isdigit((unsigned char)line[0]) || line[0] == '-' || line[0] == '.'))
        {
            first = false; // header
            continue;
        }
        first = false;
        rows.push_back(ParseList(line));
    }
    return rows;
}

// --------------------------------------------------------------------------- //
static const uint32_t NS = 3;
static const char* kSliceName[NS] = {"eMBB", "URLLC", "mMTC"};
static const uint32_t kHdr = 20 + 8 + 12; // IPv4 + UDP + SeqTsHeader
static const uint32_t kMinPkt = kHdr;     // smallest IP packet a source can emit

struct Global
{
    NdjsonSock sock;
    bool online{true};
    uint32_t uePerSlice{4};
    uint32_t epochs{10};
    double epochMs{1000.0};
    double binMs{1.0};
    double coreDelayMs{1.0};
    double t0{0.1};      // s, start of epoch 0 at the gNB
    bool verbose{false};
    std::string enforce{"mask"};
    std::vector<uint16_t> port{5000, 5001, 5002};
    std::vector<double> pdbMs{300.0, 10.0, 300.0};
    std::vector<double> maxPkt{1400.0, 1400.0, 1400.0};
    std::vector<double> capBitsDefault{1000.0, 1000.0, 1000.0}; // shaper: bit per PRB per slot
    double slotMs{1.0};
    uint32_t minPrb{4}; // per-slice quota floor (PRBs): <= 3 PRBs at MCS 0 deadlocks, see SetMask

    // schedule
    std::vector<float> sched; // [bins x cols]
    uint32_t cols{0};         // NS (aggregate) or NS*U (per-UE)
    uint64_t nBins{0};
    bool perUe{false};
    std::vector<double> theta{20.0, 5.0, 2.0}; // Mbps if no schedule
    // budgets (offline)
    std::vector<std::vector<double>> budgets;
    std::ofstream kpiOut;

    // topology handles
    std::vector<Ptr<NrMacSchedulerNs3>> sched3;
    std::vector<Ptr<NetDevice>> sliceLink;
    std::vector<std::vector<Ptr<Socket>>> txSock; // [s][u] on host s
    uint32_t rbPerBwp{0};
    uint32_t rbPerRbg{1};
    uint32_t gnbNodeId{0};
    std::map<uint32_t, int> addrSlice; // UE IPv4 -> slice
    std::map<uint16_t, int> rntiSlice; // RNTI -> slice
    std::vector<double> rlcBuf{512 * 1024.0, 32 * 1024.0, 512 * 1024.0};
    uint32_t rlcHooks{0};

    // source state
    std::vector<std::vector<double>> carry; // [s][stream]
    std::vector<uint32_t> rr, seq;

    // per-epoch tables (index = epoch)
    std::vector<std::vector<double>> schedB, srcB, srcPk, rcvdPkBySend;
    // current-epoch accumulators, reset at EndEpoch
    std::vector<double> pdcpB, rxB, rxPdbB, rxPk, dropB, usedReg, availSym, tbB, mcsSum, nTb;
    std::vector<std::vector<double>> delays;
    std::vector<double> quota, prbReq;
} g;

static inline int64_t
EpochOfGnbTime(double t)
{
    return (int64_t)std::floor((t - g.t0) * 1000.0 / g.epochMs + 1e-12);
}

// ---------------------------------------------------------------- sources -- //
static void
SendPkt(uint32_t s, uint32_t u, uint32_t ipBytes)
{
    Ptr<Packet> p = Create<Packet>(ipBytes - kHdr);
    SeqTsHeader h; // timestamp = Now
    h.SetSeq(g.seq[s]++);
    p->AddHeader(h);
    g.txSock[s][u]->Send(p);
}

static void
SourceBin(uint32_t s, uint64_t b)
{
    const uint64_t binsPerEpoch = (uint64_t)std::llround(g.epochMs / g.binMs);
    const uint64_t epoch = b / binsPerEpoch;
    if (epoch >= g.epochs)
        return;
    const uint64_t row = g.nBins ? (b % g.nBins) : 0;
    const uint32_t nStreams = g.perUe ? g.uePerSlice : 1;
    std::vector<std::pair<uint32_t, uint32_t>> pk; // (ue, ipBytes)
    for (uint32_t st = 0; st < nStreams; ++st)
    {
        double bytes;
        if (g.nBins == 0)
            bytes = g.theta[s] * 1e6 / 8.0 * g.binMs / 1000.0;
        else
            bytes = g.sched[row * g.cols + (g.perUe ? s * g.uePerSlice + st : s)];
        g.schedB[s][epoch] += bytes;
        double& c = g.carry[s][st];
        c += bytes;
        while (c >= kMinPkt)
        {
            uint32_t sz = (uint32_t)std::min(g.maxPkt[s], std::floor(c));
            uint32_t ue = g.perUe ? st : (g.rr[s]++ % g.uePerSlice);
            pk.emplace_back(ue, sz);
            c -= sz;
        }
    }
    const double dt = g.binMs / 1000.0 / std::max<size_t>(pk.size(), 1);
    for (size_t j = 0; j < pk.size(); ++j)
    {
        g.srcB[s][epoch] += pk[j].second;
        g.srcPk[s][epoch] += 1;
        if (j == 0)
            SendPkt(s, pk[j].first, pk[j].second);
        else
            Simulator::Schedule(Seconds(j * dt), &SendPkt, s, pk[j].first, pk[j].second);
    }
    Simulator::Schedule(Seconds(g.binMs / 1000.0), &SourceBin, s, b + 1);
}

// -------------------------------------------------------------- receivers -- //
static void
OnUeRx(uint32_t s, Ptr<Socket> sock)
{
    Ptr<Packet> p;
    while ((p = sock->Recv()))
    {
        if (p->GetSize() < 12)
            continue;
        SeqTsHeader h;
        p->RemoveHeader(h);
        const double ip = p->GetSize() + kHdr;
        const double now = Simulator::Now().GetSeconds();
        const double dms = (Simulator::Now() - h.GetTs()).GetSeconds() * 1000.0;
        int64_t txEp = EpochOfGnbTime(h.GetTs().GetSeconds() + g.coreDelayMs / 1000.0);
        if (txEp >= 0 && txEp < (int64_t)g.epochs)
            g.rcvdPkBySend[s][txEp] += 1;
        if (EpochOfGnbTime(now) >= 0)
        {
            g.rxB[s] += ip;
            g.rxPk[s] += 1;
            if (dms <= g.pdbMs[s])
                g.rxPdbB[s] += ip;
            g.delays[s].push_back(dms);
        }
    }
}

// ------------------------------------------------- gNB-side counters (RAN) -- //
static void
OnS1uRx(Ptr<Packet> p)
{
    Ipv4Header h;
    if (p->PeekHeader(h) == 0)
        return;
    auto it = g.addrSlice.find(h.GetDestination().Get());
    if (it != g.addrSlice.end() && EpochOfGnbTime(Simulator::Now().GetSeconds()) >= 0)
        g.pdcpB[it->second] += p->GetSize();
}

static void
OnRlcDrop(std::string ctx, Ptr<const Packet> p)
{
    auto a = ctx.find("UeMap/");
    if (a == std::string::npos)
        return;
    a += 6;
    auto b = ctx.find('/', a);
    uint16_t rnti = uint16_t(std::stoul(ctx.substr(a, b - a)));
    auto it = g.rntiSlice.find(rnti);
    if (it != g.rntiSlice.end())
        g.dropB[it->second] += p->GetSize();
}

// MAC: every NEW transport block (retx are not traced, nr-gnb-mac.cc:1279-1292)
static void
OnDlSched(NrSchedulingCallbackInfo i)
{
    if (i.m_bwpId < NS)
    {
        g.tbB[i.m_bwpId] += i.m_tbSize;
        g.mcsSum[i.m_bwpId] += i.m_mcs;
        g.nTb[i.m_bwpId] += 1;
    }
}

static void
OnSlotData(const SfnSf&, uint32_t, uint32_t usedReg, uint32_t, uint32_t, uint32_t availSym,
           uint16_t bwpId, uint16_t)
{
    if (bwpId < NS)
    {
        g.usedReg[bwpId] += usedReg;
        g.availSym[bwpId] += availSym;
    }
}

// ------------------------------------------------------- PRB enforcement -- //
static void
SetMask(uint32_t s, uint32_t nRbgOn)
{
    uint32_t nRbg = g.rbPerBwp / g.rbPerRbg;
    // >= 1 RBG (the OFDMA scheduler asserts on an empty set, nr-mac-scheduler-ofdma.cc:368) and
    // >= minPrb (a slice whose TBS at the starting MCS is < 10 B is never scheduled, so it
    // never gets a CQI: nr-mac-scheduler-ofdma.cc:438 reaps it -- a starvation deadlock).
    uint32_t floorRbg = std::max<uint32_t>(1, (g.minPrb + g.rbPerRbg - 1) / g.rbPerRbg);
    nRbgOn = std::max<uint32_t>(floorRbg, std::min(nRbgOn, nRbg));
    std::vector<bool> mask(nRbg, false);
    for (uint32_t i = 0; i < nRbgOn; ++i)
        mask[i] = true;
    g.sched3[s]->SetDlNotchedRbgMask(mask);
    g.quota[s] = double(nRbgOn * g.rbPerRbg);
}

static void
ApplyAllocation(const std::vector<double>& prb, const std::vector<double>& capBitsIn)
{
    for (uint32_t s = 0; s < NS; ++s)
    {
        double q = s < prb.size() ? prb[s] : 0.0;
        g.prbReq[s] = q;
        if (g.enforce == "mask")
        {
            SetMask(s, (uint32_t)std::lround(q / g.rbPerRbg));
        }
        else if (g.enforce == "shaper")
        {
            double cap = s < capBitsIn.size() ? capBitsIn[s] : g.capBitsDefault[s];
            double bps = std::max(q * cap / (g.slotMs / 1000.0), 1e5);
            DynamicCast<PointToPointNetDevice>(g.sliceLink[s])->SetDataRate(DataRate(uint64_t(bps)));
            g.quota[s] = q; // masks stay at the static equal split set in Hello()
        }
        else
        {
            g.quota[s] = double(g.rbPerBwp);
        }
    }
}

// ------------------------------------------------------------ epoch loop -- //
static std::string
Arr(const std::vector<double>& v)
{
    std::ostringstream o;
    o << std::setprecision(10) << "[";
    for (size_t i = 0; i < v.size(); ++i)
        o << v[i] << (i + 1 < v.size() ? "," : "");
    o << "]";
    return o.str();
}

static double
Pct(std::vector<double> v, double q)
{
    if (v.empty())
        return -1.0;
    size_t k = (size_t)std::min<double>(v.size() - 1, std::floor(q * (v.size() - 1) + 0.5));
    std::nth_element(v.begin(), v.begin() + k, v.end());
    return v[k];
}

static void EndEpoch(uint32_t epoch);

static void
RunEpoch(uint32_t epoch)
{
    std::vector<double> prb, cap;
    if (g.online)
    {
        std::string msg = g.sock.RecvLine();
        if (msg.empty() || msg.find("\"done\"") != std::string::npos)
        {
            Simulator::Stop();
            return;
        }
        prb = JsonArray(msg, "prb");
        cap = JsonArray(msg, "cap_bits");
    }
    else if (!g.budgets.empty())
    {
        prb = g.budgets[std::min<size_t>(epoch, g.budgets.size() - 1)];
    }
    else
    {
        prb.assign(NS, double(g.rbPerBwp) / NS);
    }
    ApplyAllocation(prb, cap);
    Simulator::Schedule(MilliSeconds(g.epochMs), &EndEpoch, epoch);
}

static void
EndEpoch(uint32_t epoch)
{
    std::vector<double> sched(NS), src(NS), srcPk(NS), pdcp(NS), rx(NS), rxPdb(NS), rxPk(NS), drop(NS),
        dMean(NS), d50(NS), d95(NS), d99(NS), dMax(NS), inPdb(NS), lostPrev(NS), sentPrev(NS), rbUsed(NS), tbB(NS), mcsMean(NS);
    for (uint32_t s = 0; s < NS; ++s)
    {
        sched[s] = g.schedB[s][epoch];
        src[s] = g.srcB[s][epoch];
        srcPk[s] = g.srcPk[s][epoch];
        pdcp[s] = g.pdcpB[s];
        rx[s] = g.rxB[s];
        rxPdb[s] = g.rxPdbB[s];
        rxPk[s] = g.rxPk[s];
        drop[s] = g.dropB[s];
        const auto& d = g.delays[s];
        double sum = 0, mx = 0, ok = 0;
        for (double x : d)
        {
            sum += x;
            mx = std::max(mx, x);
            ok += (x <= g.pdbMs[s]);
        }
        dMean[s] = d.empty() ? -1.0 : sum / d.size();
        dMax[s] = d.empty() ? -1.0 : mx;
        inPdb[s] = d.empty() ? -1.0 : ok / d.size();
        d50[s] = Pct(d, 0.50);
        d95[s] = Pct(d, 0.95);
        d99[s] = Pct(d, 0.99);
        // loss of the PREVIOUS epoch's packets, final once a whole epoch (>> PDB) has passed
        if (epoch >= 1)
        {
            sentPrev[s] = g.srcPk[s][epoch - 1];
            lostPrev[s] = std::max(0.0, sentPrev[s] - g.rcvdPkBySend[s][epoch - 1]);
        }
        rbUsed[s] = g.availSym[s] > 0 ? g.usedReg[s] / g.availSym[s] : 0.0;
        tbB[s] = g.tbB[s];
        mcsMean[s] = g.nTb[s] > 0 ? g.mcsSum[s] / g.nTb[s] : -1.0;
    }
    std::ostringstream o;
    o << "{\"type\":\"kpi\",\"epoch\":" << epoch
      << ",\"t_end_s\":" << std::setprecision(10) << Simulator::Now().GetSeconds()
      << ",\"prb_req\":" << Arr(g.prbReq) << ",\"prb_quota\":" << Arr(g.quota)
      << ",\"prb_used_mean\":" << Arr(rbUsed) << ",\"sched_bytes\":" << Arr(sched)
      << ",\"src_bytes\":" << Arr(src) << ",\"src_pkts\":" << Arr(srcPk)
      << ",\"pdcp_sdu_bytes\":" << Arr(pdcp) << ",\"delivered_bytes\":" << Arr(rx)
      << ",\"delivered_pdb_bytes\":" << Arr(rxPdb) << ",\"rx_pkts\":" << Arr(rxPk)
      << ",\"rlc_drop_bytes\":" << Arr(drop) << ",\"delay_mean_ms\":" << Arr(dMean)
      << ",\"delay_p50_ms\":" << Arr(d50) << ",\"delay_p95_ms\":" << Arr(d95)
      << ",\"delay_p99_ms\":" << Arr(d99) << ",\"delay_max_ms\":" << Arr(dMax)
      << ",\"within_pdb_frac\":" << Arr(inPdb) << ",\"lost_prev_pkts\":" << Arr(lostPrev)
      << ",\"sent_prev_pkts\":" << Arr(sentPrev) << ",\"tb_new_bytes\":" << Arr(tbB)
      << ",\"mcs_mean\":" << Arr(mcsMean) << "}";
    g.sock.SendLine(o.str());
    if (g.verbose)
        std::cout << o.str() << std::endl;
    if (g.kpiOut.is_open())
        for (uint32_t s = 0; s < NS; ++s)
            g.kpiOut << epoch << "," << kSliceName[s] << "," << g.prbReq[s] << "," << g.quota[s]
                     << "," << rbUsed[s] << "," << sched[s] << "," << src[s] << "," << pdcp[s]
                     << "," << rx[s] << "," << rxPdb[s] << "," << drop[s] << "," << srcPk[s]
                     << "," << rxPk[s] << "," << dMean[s] << "," << d50[s] << "," << d95[s]
                     << "," << d99[s] << "," << dMax[s] << "," << inPdb[s] << ","
                     << lostPrev[s] << "," << sentPrev[s] << "," << tbB[s] << "," << mcsMean[s]
                     << "\n";
    for (uint32_t s = 0; s < NS; ++s)
    {
        g.pdcpB[s] = g.rxB[s] = g.rxPdbB[s] = g.rxPk[s] = g.dropB[s] = 0.0;
        g.usedReg[s] = g.availSym[s] = g.tbB[s] = g.mcsSum[s] = g.nTb[s] = 0.0;
        g.delays[s].clear();
    }
    if (epoch + 1 < g.epochs)
        Simulator::ScheduleNow(&RunEpoch, epoch + 1);
    else
        Simulator::Stop();
}

static std::string g_helloExtra;

// N_RB is known only after PHY initialisation, so the hello (which carries the budget
// N_RB) is sent from inside the simulation, just before epoch 0.
static void
Hello(NetDeviceContainer gnbDev)
{
    g.rbPerBwp = NrHelper::GetGnbPhy(gnbDev.Get(0), 0)->GetRbNum();
    g.rbPerRbg = NrHelper::GetGnbMac(gnbDev.Get(0), 0)->GetNumRbPerRbg();
    for (uint32_t w = 1; w < NS; ++w)
        NS_ABORT_MSG_IF(NrHelper::GetGnbPhy(gnbDev.Get(0), w)->GetRbNum() != g.rbPerBwp,
                        "BWPs differ in RB count");
    NS_ABORT_MSG_IF(g.rbPerBwp == 0, "RB count not initialised yet");
    if (g.enforce == "shaper")
        for (uint32_t s = 0; s < NS; ++s)
            SetMask(s, g.rbPerBwp / g.rbPerRbg / NS); // the old static 3-BWP partition
    // RNTI -> slice is known now too
    std::ostringstream h;
    h << "{\"type\":\"hello\",\"role\":\"ns3\",\"scenario\":\"vslice-nr-5g\",\"slices\":" << NS
      << ",\"slice_names\":[\"eMBB\",\"URLLC\",\"mMTC\"],\"rb_total\":" << g.rbPerBwp
      << ",\"rb_per_rbg\":" << g.rbPerRbg << ",\"enforce\":\"" << g.enforce
      << "\",\"measurement\":\"pdcp_sdu_bytes\",\"ue_per_slice\":" << g.uePerSlice
      << ",\"epoch_ms\":" << g.epochMs << ",\"bin_ms\":" << g.binMs
      << ",\"sched_bins\":" << g.nBins << ",\"per_ue_schedule\":" << (g.perUe ? 1 : 0)
      << ",\"pdb_ms\":" << Arr(g.pdbMs) << g_helloExtra << "}";
    g.sock.SendLine(h.str());
    if (g.online)
        g.sock.RecvLine(); // config
    if (g.verbose)
        std::cout << h.str() << std::endl;
}

static void
LateHooks(NetDeviceContainer ueDevs)
{
    for (uint32_t i = 0; i < ueDevs.GetN(); ++i)
    {
        auto ue = DynamicCast<NrUeNetDevice>(ueDevs.Get(i));
        g.rntiSlice[ue->GetRrc()->GetRnti()] = int(i / g.uePerSlice);
    }
    const std::string base = "/NodeList/" + std::to_string(g.gnbNodeId) +
                             "/DeviceList/*/NrGnbRrc/UeMap/";
    // per-slice RLC UM buffer (bytes per RLC entity)
    for (const auto& kv : g.rntiSlice)
    {
        Config::MatchContainer m = Config::LookupMatches(
            base + std::to_string(kv.first) + "/DataRadioBearerMap/*/NrRlc");
        for (uint32_t j = 0; j < m.GetN(); ++j)
        {
            Ptr<NrRlcUm> um = DynamicCast<NrRlcUm>(m.Get(j));
            if (um)
                um->SetAttribute("MaxTxBufferSize", UintegerValue((uint32_t)g.rlcBuf[kv.second]));
        }
    }
    // (LookupMatches resolves objects, not trace sources: count the RLC entities instead)
    g.rlcHooks = Config::LookupMatches(base + "*/DataRadioBearerMap/*/NrRlc").GetN();
    NS_ABORT_MSG_IF(!Config::ConnectFailSafe(base + "*/DataRadioBearerMap/*/NrRlc/TxDrop",
                                             MakeCallback(&OnRlcDrop)),
                    "could not hook RLC TxDrop");
    if (g.verbose)
        std::cout << "[ns-3] RLC TxDrop hooks: " << g.rlcHooks
                  << ", RNTIs mapped: " << g.rntiSlice.size() << std::endl;
}

// --------------------------------------------------------------------------- //
int
main(int argc, char* argv[])
{
    std::string host = "127.0.0.1";
    uint16_t port = 5005;
    uint32_t rngRun = 1;
    double bwHz = 20e6; // width of EACH slice BWP (== the whole cell)
    double fc = 3.5e9;
    double txPowerDbm = 35.0;
    double ueSpeed = 3.0;
    uint32_t numerology = 0;
    uint32_t mcsTable = 2;
    uint32_t startMcsDl = 0;
    double warmupMs = 100.0;
    double discRadius = 60.0;
    bool allowManyUes = false;
    bool sharedBand = true;
    std::string scheduleFile, budgetFile, kpiFile;
    std::string pdb = "300,10,300", maxPkt = "1400,1400,1400", rlcBuf = "524288,32768,524288";
    std::string theta = "20,5,2", capBits = "1000,1000,1000";
    std::string scheduler = "ns3::NrMacSchedulerOfdmaPF";

    CommandLine cmd(__FILE__);
    cmd.AddValue("host", "controller host", host);
    cmd.AddValue("port", "controller port (0 = offline: --budgetFile)", port);
    cmd.AddValue("epochs", "number of epochs", g.epochs);
    cmd.AddValue("epochMs", "epoch length (ms); multiple of binMs", g.epochMs);
    cmd.AddValue("uePerSlice", "UEs per slice (<= 8 unless --allowManyUes)", g.uePerSlice);
    cmd.AddValue("allowManyUes", "permit uePerSlice > 8 (known nr crashes beyond ~24 UEs)",
                 allowManyUes);
    cmd.AddValue("rngRun", "RNG run (CRN: channel, mobility, placement)", rngRun);
    cmd.AddValue("scheduleFile", "CSV byte schedule (see header comment)", scheduleFile);
    cmd.AddValue("binMs", "schedule bin width (ms)", g.binMs);
    cmd.AddValue("theta", "if no schedule: per-slice constant rate, Mbps (eMBB,URLLC,mMTC)", theta);
    cmd.AddValue("maxPkt", "per-slice max IP packet size (bytes)", maxPkt);
    cmd.AddValue("pdb", "per-slice packet delay budget for the KPI (ms)", pdb);
    cmd.AddValue("rlcBuf", "per-slice RLC UM buffer per UE (bytes)", rlcBuf);
    cmd.AddValue("budgetFile", "offline mode: CSV of per-epoch PRB quotas", budgetFile);
    cmd.AddValue("kpiFile", "write per-epoch per-slice KPIs to this CSV", kpiFile);
    cmd.AddValue("enforce", "mask | shaper | none", g.enforce);
    cmd.AddValue("capBits", "shaper only: default bit per PRB per slot (per slice)", capBits);
    cmd.AddValue("bandwidth", "width of EACH slice BWP == cell width (Hz)", bwHz);
    cmd.AddValue("freq", "centre frequency of BWP 0 (Hz)", fc);
    cmd.AddValue("sharedBand",
                 "1: one band of 3W holding the 3 W-wide BWPs (one channel model, ~1.6x "
                 "faster); 0: three separate W-wide bands",
                 sharedBand);
    cmd.AddValue("txPower", "gNB total tx power (dBm)", txPowerDbm);
    cmd.AddValue("numerology", "numerology of all BWPs (0 -> 1 ms slot)", numerology);
    cmd.AddValue("minPrb", "mask: per-slice PRB floor (starvation guard, see SetMask)", g.minPrb);
    cmd.AddValue("startMcsDl", "DL MCS used before the first CQI (5G-LENA default 0)", startMcsDl);
    cmd.AddValue("mcsTable", "1 = 64QAM (EESM-IR T1), 2 = 256QAM (T2)", mcsTable);
    cmd.AddValue("ueSpeed", "eMBB UE speed (m/s)", ueSpeed);
    cmd.AddValue("discRadius", "UE drop radius around the gNB (m)", discRadius);
    cmd.AddValue("warmupMs", "time before epoch 0 (attach, bearer setup)", warmupMs);
    cmd.AddValue("coreDelayMs", "remote host -> PGW link delay (ms)", g.coreDelayMs);
    cmd.AddValue("scheduler", "NR MAC scheduler TypeId", scheduler);
    cmd.AddValue("verbose", "print hello/KPIs to stdout", g.verbose);
    cmd.Parse(argc, argv);

    NS_ABORT_MSG_IF(g.uePerSlice == 0, "uePerSlice must be >= 1");
    NS_ABORT_MSG_IF(g.uePerSlice > 8 && !allowManyUes,
                    "uePerSlice > 8: 5G-LENA crashed beyond ~24-36 UEs in this harness");
    NS_ABORT_MSG_IF(std::fmod(g.epochMs, g.binMs) > 1e-9, "epochMs must be a multiple of binMs");
    NS_ABORT_MSG_IF(g.coreDelayMs * 2 >= warmupMs, "warmupMs must exceed 2 x coreDelayMs");
    g.online = port != 0;
    g.pdbMs = ParseList(pdb);
    g.maxPkt = ParseList(maxPkt);
    g.rlcBuf = ParseList(rlcBuf);
    g.theta = ParseList(theta);
    g.capBitsDefault = ParseList(capBits);
    for (auto* v : {&g.pdbMs, &g.maxPkt, &g.rlcBuf, &g.theta, &g.capBitsDefault})
        NS_ABORT_MSG_IF(v->size() != NS, "per-slice lists need exactly 3 values");
    for (double m : g.maxPkt)
        NS_ABORT_MSG_IF(m < kMinPkt || m > 1900, "maxPkt must be in [40, 1900] bytes");
    g.t0 = warmupMs / 1000.0;

    RngSeedManager::SetSeed(1);
    RngSeedManager::SetRun(rngRun);

    if (!scheduleFile.empty())
    {
        auto rows = ReadCsv(scheduleFile);
        NS_ABORT_MSG_IF(rows.empty(), "empty schedule");
        uint32_t c = rows[0].size() - 1;
        NS_ABORT_MSG_IF(c != NS && c != NS * g.uePerSlice,
                        "schedule needs 1+" << NS << " or 1+" << NS * g.uePerSlice << " columns, got "
                                            << rows[0].size());
        g.cols = c;
        g.perUe = (c == NS * g.uePerSlice) && g.uePerSlice > 1;
        g.nBins = rows.size();
        g.sched.reserve(g.nBins * c);
        for (auto& r : rows)
        {
            NS_ABORT_MSG_IF(r.size() != c + 1, "ragged schedule row");
            for (uint32_t j = 1; j <= c; ++j)
                g.sched.push_back((float)r[j]);
        }
        double need = g.epochs * g.epochMs / g.binMs;
        if (g.nBins < need)
            std::cerr << "[ns-3] WARNING: schedule has " << g.nBins << " bins < " << need
                      << " needed; it will wrap around" << std::endl;
    }
    if (!budgetFile.empty())
        g.budgets = ReadCsv(budgetFile);
    NS_ABORT_MSG_IF(!g.online && g.budgets.empty() && g.enforce != "none",
                    "offline mode needs --budgetFile (or --enforce=none)");
    if (!kpiFile.empty())
    {
        g.kpiOut.open(kpiFile);
        g.kpiOut << std::setprecision(10)
                 << "epoch,slice,prb_req,prb_quota,prb_used_mean,sched_bytes,src_bytes,"
                    "pdcp_sdu_bytes,delivered_bytes,delivered_pdb_bytes,rlc_drop_bytes,src_pkts,"
                    "rx_pkts,delay_mean_ms,delay_p50_ms,delay_p95_ms,delay_p99_ms,delay_max_ms,"
                    "within_pdb_frac,lost_prev_pkts,sent_prev_pkts,tb_new_bytes,mcs_mean\n";
    }

    // per-epoch tables
    auto zeros2 = [](uint32_t n) { return std::vector<std::vector<double>>(NS, std::vector<double>(n, 0.0)); };
    g.schedB = zeros2(g.epochs + 1);
    g.srcB = zeros2(g.epochs + 1);
    g.srcPk = zeros2(g.epochs + 1);
    g.rcvdPkBySend = zeros2(g.epochs + 1);
    for (auto* v : {&g.pdcpB, &g.rxB, &g.rxPdbB, &g.rxPk, &g.dropB, &g.usedReg, &g.availSym, &g.tbB, &g.mcsSum, &g.nTb,
                    &g.quota, &g.prbReq})
        v->assign(NS, 0.0);
    g.delays.assign(NS, {});
    g.carry.assign(NS, std::vector<double>(g.uePerSlice, 0.0));
    g.rr.assign(NS, 0);
    g.seq.assign(NS, 0);

    // RLC: drop only on a real buffer overflow (EnablePdcpDiscarding fires TxDrop but still
    // enqueues the SDU, nr-rlc-um.cc:103-131, which would double count).
    Config::SetDefault("ns3::NrRlcUm::MaxTxBufferSize", UintegerValue(10 * 1024 * 1024));
    Config::SetDefault("ns3::NrRlcUm::EnablePdcpDiscarding", BooleanValue(false));
    Config::SetDefault("ns3::NrAmc::ErrorModelType",
                       TypeIdValue(TypeId::LookupByName(mcsTable == 1 ? "ns3::NrEesmIrT1"
                                                                      : "ns3::NrEesmIrT2")));
    Config::SetDefault("ns3::NrAmc::AmcModel", EnumValue(NrAmc::ShannonModel));
    Config::SetDefault("ns3::NrMacSchedulerNs3::StartingMcsDl", UintegerValue(startMcsDl));

    if (g.online && !g.sock.Connect(host, port))
        NS_FATAL_ERROR("cannot reach controller at " << host << ":" << port);

    Ptr<NrPointToPointEpcHelper> epc = CreateObject<NrPointToPointEpcHelper>();
    Ptr<NrHelper> nr = CreateObject<NrHelper>();
    nr->SetEpcHelper(epc);
    nr->SetSchedulerTypeId(TypeId::LookupByName(scheduler));
    nr->SetGnbPhyAttribute("PowerAllocationType",
                           EnumValue(NrSpectrumValueHelper::UNIFORM_POWER_ALLOCATION_BW));

    NodeContainer gnb, ueE, ueU, ueM;
    gnb.Create(1);
    ueE.Create(g.uePerSlice);
    ueU.Create(g.uePerSlice);
    ueM.Create(g.uePerSlice);
    NodeContainer ues(ueE, ueU, ueM); // slice-major: index i -> slice i / uePerSlice
    g.gnbNodeId = gnb.Get(0)->GetId();

    MobilityHelper mg;
    mg.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    Ptr<ListPositionAllocator> gp = CreateObject<ListPositionAllocator>();
    gp->Add(Vector(0, 0, 25));
    mg.SetPositionAllocator(gp);
    mg.Install(gnb);
    Ptr<UniformDiscPositionAllocator> disc = CreateObject<UniformDiscPositionAllocator>();
    disc->SetX(0);
    disc->SetY(0);
    disc->SetRho(discRadius);
    disc->SetZ(1.5);
    MobilityHelper ms;
    ms.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    ms.SetPositionAllocator(disc);
    ms.Install(ueU);
    ms.Install(ueM);
    MobilityHelper mw; // eMBB users walk
    mw.SetPositionAllocator(disc);
    mw.SetMobilityModel("ns3::RandomWalk2dMobilityModel",
                        "Mode", StringValue("Time"),
                        "Time", StringValue("0.5s"),
                        "Speed", StringValue("ns3::ConstantRandomVariable[Constant=" +
                                             std::to_string(ueSpeed) + "]"),
                        "Bounds", RectangleValue(Rectangle(-discRadius, discRadius,
                                                           -discRadius, discRadius)));
    mw.Install(ueE);

    nr->SetGnbAntennaAttribute("NumRows", UintegerValue(2));
    nr->SetGnbAntennaAttribute("NumColumns", UintegerValue(4));
    nr->SetGnbAntennaAttribute("AntennaElement", PointerValue(CreateObject<ThreeGppAntennaModel>()));
    nr->SetUeAntennaAttribute("NumRows", UintegerValue(1));
    nr->SetUeAntennaAttribute("NumColumns", UintegerValue(2));
    nr->SetUeAntennaAttribute("AntennaElement", PointerValue(CreateObject<IsotropicAntennaModel>()));

    // Three adjacent W-wide BWPs (BWP s centred at fc + s*W), one per slice.  sharedBand=1
    // puts them in ONE operation band of 3W (one CC, 3 BWPs), so one 3GPP channel model
    // serves all three; sharedBand=0 makes three separate W-wide bands.  The BWPs are the
    // same in both layouts; only the channel-model bookkeeping (and run time) differs.
    CcBwpCreator ccbwp;
    std::vector<OperationBandInfo> bands;
    if (sharedBand)
    {
        CcBwpCreator::SimpleOperationBandConf conf(fc + bwHz, NS * bwHz, 1);
        conf.m_numBwp = NS;
        bands.push_back(ccbwp.CreateOperationBandContiguousCc(conf));
    }
    else
    {
        for (uint32_t s = 0; s < NS; ++s)
        {
            CcBwpCreator::SimpleOperationBandConf conf(fc + s * bwHz, bwHz, 1);
            conf.m_numBwp = 1;
            bands.push_back(ccbwp.CreateOperationBandContiguousCc(conf));
        }
    }
    Ptr<NrChannelHelper> ch = CreateObject<NrChannelHelper>();
    ch->ConfigureFactories("UMi", "Default", "ThreeGpp");
    ch->SetPathlossAttribute("ShadowingEnabled", BooleanValue(true));
    std::vector<std::reference_wrapper<OperationBandInfo>> bandRefs(bands.begin(), bands.end());
    ch->AssignChannelsToBands(bandRefs);
    BandwidthPartInfoPtrVector allBwps = CcBwpCreator::GetAllBwps(bandRefs);

    // 5QI -> BWP (== slice).  eMBB 5QI 8, URLLC 5QI 83, mMTC 5QI 9 (also the default bearer,
    // which carries no traffic here).
    nr->SetGnbBwpManagerAlgorithmAttribute("NGBR_VIDEO_TCP_PREMIUM", UintegerValue(0));
    nr->SetGnbBwpManagerAlgorithmAttribute("DGBR_DISCRETE_AUT_LARGE", UintegerValue(1));
    nr->SetGnbBwpManagerAlgorithmAttribute("NGBR_VIDEO_TCP_DEFAULT", UintegerValue(2));
    nr->SetUeBwpManagerAlgorithmAttribute("NGBR_VIDEO_TCP_PREMIUM", UintegerValue(0));
    nr->SetUeBwpManagerAlgorithmAttribute("DGBR_DISCRETE_AUT_LARGE", UintegerValue(1));
    nr->SetUeBwpManagerAlgorithmAttribute("NGBR_VIDEO_TCP_DEFAULT", UintegerValue(2));

    NetDeviceContainer gnbDev = nr->InstallGnbDevice(gnb, allBwps);
    NetDeviceContainer devE = nr->InstallUeDevice(ueE, allBwps);
    NetDeviceContainer devU = nr->InstallUeDevice(ueU, allBwps);
    NetDeviceContainer devM = nr->InstallUeDevice(ueM, allBwps);
    NetDeviceContainer ueDev;
    ueDev.Add(devE);
    ueDev.Add(devU);
    ueDev.Add(devM);

    for (uint32_t w = 0; w < NS; ++w)
    {
        auto phy = NrHelper::GetGnbPhy(gnbDev.Get(0), w);
        phy->SetAttribute("Numerology", UintegerValue(numerology));
        // Same PSD as ONE carrier of width W at txPower (UniformPowerAllocBw): at most
        // sum_s n_s <= N_RB RBs are ever active across the three BWPs.
        phy->SetAttribute("TxPower", DoubleValue(txPowerDbm));
        g.sched3.push_back(DynamicCast<NrMacSchedulerNs3>(NrHelper::GetScheduler(gnbDev.Get(0), w)));
        NS_ABORT_MSG_IF(!g.sched3.back(), "scheduler of BWP " << w << " is not NrMacSchedulerNs3");
    }
    g.slotMs = 1.0 / std::pow(2.0, numerology);

    // Core: one remote host per slice.  10 Gb/s (mask) or the shaped bottleneck (shaper).
    InternetStackHelper internet;
    Ptr<Node> pgw = epc->GetPgwNode();
    NodeContainer hosts;
    hosts.Create(NS);
    internet.Install(hosts);
    Ipv4AddressHelper ipv4;
    g.sliceLink.resize(NS);
    for (uint32_t s = 0; s < NS; ++s)
    {
        PointToPointHelper p2p;
        p2p.SetDeviceAttribute("DataRate", DataRateValue(DataRate("10Gb/s")));
        p2p.SetDeviceAttribute("Mtu", UintegerValue(2500));
        p2p.SetChannelAttribute("Delay", TimeValue(Seconds(g.coreDelayMs / 1000.0)));
        NetDeviceContainer d = p2p.Install(hosts.Get(s), pgw);
        std::ostringstream net;
        net << "10." << (s + 1) << ".0.0";
        ipv4.SetBase(Ipv4Address(net.str().c_str()), "255.255.255.0");
        ipv4.Assign(d);
        g.sliceLink[s] = d.Get(0);
        Ipv4StaticRoutingHelper srh;
        srh.GetStaticRouting(hosts.Get(s)->GetObject<Ipv4>())
            ->AddNetworkRouteTo(Ipv4Address("7.0.0.0"), Ipv4Mask("255.0.0.0"), 1);
    }

    internet.Install(ues);
    Ipv4InterfaceContainer ueIp = epc->AssignUeIpv4Address(ueDev);
    for (uint32_t i = 0; i < ues.GetN(); ++i)
    {
        Ipv4StaticRoutingHelper().GetStaticRouting(ues.Get(i)->GetObject<Ipv4>())
            ->SetDefaultRoute(epc->GetUeDefaultGatewayAddress(), 1);
        g.addrSlice[ueIp.GetAddress(i).Get()] = int(i / g.uePerSlice);
    }
    nr->AttachToClosestGnb(ueDev, gnbDev);

    NrEpsBearer bearers[NS] = {NrEpsBearer(NrEpsBearer::NGBR_VIDEO_TCP_PREMIUM),
                              NrEpsBearer(NrEpsBearer::DGBR_DISCRETE_AUT_LARGE),
                              NrEpsBearer(NrEpsBearer::NGBR_VIDEO_TCP_DEFAULT)};
    NetDeviceContainer sliceDev[NS] = {devE, devU, devM};
    for (uint32_t s = 0; s < NS; ++s)
    {
        Ptr<NrEpcTft> tft = Create<NrEpcTft>();
        NrEpcTft::PacketFilter pf;
        pf.localPortStart = g.port[s];
        pf.localPortEnd = g.port[s];
        tft->Add(pf);
        for (uint32_t u = 0; u < sliceDev[s].GetN(); ++u)
            nr->ActivateDedicatedEpsBearer(sliceDev[s].Get(u), bearers[s], tft);
    }

    // Sinks (UE sockets) and sources (host sockets).  Sources start at t0 - coreDelay.
    g.txSock.assign(NS, {});
    for (uint32_t s = 0; s < NS; ++s)
        for (uint32_t u = 0; u < g.uePerSlice; ++u)
        {
            uint32_t idx = s * g.uePerSlice + u;
            Ptr<Socket> rx = Socket::CreateSocket(ues.Get(idx), UdpSocketFactory::GetTypeId());
            rx->Bind(InetSocketAddress(Ipv4Address::GetAny(), g.port[s]));
            rx->SetRecvCallback(MakeBoundCallback(&OnUeRx, s));
            Ptr<Socket> tx = Socket::CreateSocket(hosts.Get(s), UdpSocketFactory::GetTypeId());
            tx->Bind();
            tx->Connect(InetSocketAddress(ueIp.GetAddress(idx), g.port[s]));
            g.txSock[s].push_back(tx);
        }
    for (uint32_t s = 0; s < NS; ++s)
        Simulator::Schedule(Seconds(g.t0 - g.coreDelayMs / 1000.0), &SourceBin, s, (uint64_t)0);

    // gNB-side counters
    std::string s1u = "/NodeList/" + std::to_string(g.gnbNodeId) +
                      "/ApplicationList/*/$ns3::NrEpcGnbApplication/RxFromS1u";
    NS_ABORT_MSG_IF(!Config::ConnectWithoutContextFailSafe(s1u, MakeCallback(&OnS1uRx)),
                    "could not hook RxFromS1u");
    for (uint32_t w = 0; w < NS; ++w)
        NrHelper::GetGnbPhy(gnbDev.Get(0), w)
            ->TraceConnectWithoutContext("SlotDataStats", MakeCallback(&OnSlotData));
    for (uint32_t w = 0; w < NS; ++w)
        NrHelper::GetGnbMac(gnbDev.Get(0), w)
            ->TraceConnectWithoutContext("DlScheduling", MakeCallback(&OnDlSched));

    {
        std::ostringstream e;
        e << ",\"numerology\":" << numerology << ",\"bwp_hz\":" << bwHz
          << ",\"tx_power_dbm\":" << txPowerDbm << ",\"mcs_table\":" << mcsTable
          << ",\"core_delay_ms\":" << g.coreDelayMs << ",\"min_prb\":" << g.minPrb
          << ",\"start_mcs_dl\":" << startMcsDl << ",\"rng_run\":" << rngRun
          << ",\"rlc_buf\":" << Arr(g.rlcBuf) << ",\"max_pkt\":" << Arr(g.maxPkt);
        g_helloExtra = e.str();
    }
    Simulator::Schedule(Seconds(g.t0 - 0.004), &Hello, gnbDev);
    Simulator::Schedule(Seconds(g.t0 - 0.003), &LateHooks, ueDev);
    Simulator::Schedule(Seconds(g.t0), &RunEpoch, 0);
    Simulator::Stop(Seconds(g.t0 + 0.01 + g.epochs * g.epochMs / 1000.0));
    Simulator::Run();
    g.sock.SendLine("{\"type\":\"bye\"}");
    g.sock.Close();
    if (g.kpiOut.is_open())
        g.kpiOut.close();
    Simulator::Destroy();
    return 0;
}
