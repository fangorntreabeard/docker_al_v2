import matplotlib.pyplot as plt

def mean(listt):
    return sum(listt)/len(listt)

y = {}
y[1] = [0.46862745147393137, 0.401134679819913, 0.46204267366311585, 0.48187450850469515, 0.4905388672375067, 0.4897178368218432, 0.4920906677670866, 0.507952004198712, 0.46835800547541045, 0.47405382722807166, 0.4671571236919336, 0.4959904402023866, 0.4991427270630868, 0.49402873407344117, 0.4822139192492777, 0.4678910199277992, 0.5086521802268134, 0.4918737409567199, 0.5016172749208229, 0.49614832525022134, 0.5106676705357466, 0.5111556125855902, 0.5107167220530109, 0.5106689026651552, 0.5006426051923815, 0.5212668456220059, 0.4886148525996477, 0.5020865359198958, 0.5223587526231894, 0.5161290529801176, 0.5063229293111662]
y[2] = [0.4575402307346404, 0.46514011933900656, 0.4475869625597241, 0.45187562561844946, 0.4671752417349655, 0.46511836434316783, 0.47148413367534187, 0.46534968310427616, 0.4833400599449286, 0.48055712534625183, 0.4823866700353808, 0.4717229594447387, 0.4632016333972065, 0.4788886870994317, 0.48348023395034495, 0.4840559134572324, 0.46670142435836015, 0.4952349404531585, 0.48365765412501766, 0.47905132458030997, 0.48835708399574396, 0.48173978173282683, 0.48688138992789354, 0.4919834919880226, 0.49196909486442425, 0.4917908864550383, 0.5023328104272724, 0.5058053661661706, 0.49989456890463463, 0.4949490322277829, 0.49841988436482054]
y[3] = [0.4493391292729821, 0.4489917091189737, 0.4880421818665422, 0.461623446614135, 0.45634340931606715, 0.4703016517645282, 0.4529701040552301, 0.4713631201636182, 0.476423754759522, 0.4807694589302389, 0.46507806302845395, 0.48621671062068333, 0.4729501730326617, 0.4796674970471104, 0.47740313888943203, 0.47179925275255147, 0.4835571880033809, 0.49340699772427143, 0.4730346338129662, 0.4902094701451216, 0.4718510205919663, 0.4762716352551709, 0.49544818363357573, 0.4905519955498569, 0.4941732601178516, 0.49339064645048764, 0.49101765107551626, 0.49108242215816095, 0.48759788679114385, 0.4875792391221327, 0.5028610001632282]
y[4] = [0.4735578005000439, 0.4467530551117951, 0.4613569626412821, 0.4675951044092973, 0.46180099165866567, 0.4588544383612667, 0.46014397405926644, 0.47620998519569846, 0.4783235367696011, 0.46849325223735067, 0.47468546802454975, 0.4907887510255611, 0.4923921456785598, 0.48640623494693375, 0.4943275819327208, 0.4642154717728261, 0.49186248488958756, 0.48278374276512737, 0.4802496308525029, 0.48896053325167094, 0.4548295648207997, 0.4877736967971347, 0.5107628907563819, 0.4807943240253914, 0.48547471763690275, 0.5006251894618773, 0.46320925799419854, 0.49768892270226656, 0.4857146163067114, 0.4961021743791592, 0.49526529203397546]
y[5] = [0.44592128445986934, 0.4374024269100227, 0.45887671896258037, 0.4680066568793747, 0.4547959631262842, 0.45118533578879366, 0.4759242359488329, 0.47589295805741794, 0.4694436725914802, 0.47252757872013557, 0.4669285652528289, 0.459095713109134, 0.47148626088207257, 0.4851643154818173, 0.48924001683042456, 0.4912749188268541, 0.48039812911900753, 0.5006280721038022, 0.4787036231981609, 0.49589840925331413, 0.49655180778737323, 0.48305478605818397, 0.4939002089969961, 0.4853390558923265, 0.4963619528392581, 0.49599863043164183, 0.4886551759964157, 0.4961695369362081, 0.4946544377711583, 0.4950317836281922, 0.5128278778191997]
y[6] = [0.42884894051699374, 0.42327383922418965, 0.42293607816860346, 0.4369322471448067, 0.44107157643907524, 0.44472738448444843, 0.47436201803366657, 0.46227495289519155, 0.46684158097365885, 0.4700334243546341, 0.4671132527166448, 0.4540872687686631, 0.4747258672293888, 0.4805826262584381, 0.4577887334044078, 0.4800951074065647, 0.4630773953838551, 0.47685742827236577, 0.4707232582322498, 0.4758749061016462, 0.47003419403263175, 0.45862922127173394, 0.4756651101271913, 0.47666301657890914, 0.4889892987549012, 0.5003681222195852, 0.47811459885801805, 0.49996353386648257, 0.4853187936405835, 0.4920113603157366, 0.4916721520781105]
y[7] = [0.4225964267643786, 0.44991606680849017, 0.4547514888474081, 0.44145712301490286, 0.41148181308798826, 0.4512900602912159, 0.44484937235838806, 0.45360394702141793, 0.4403851161277241, 0.46165153694417305, 0.4559624171873046, 0.4566408083530057, 0.4675454954688847, 0.47074496727602166, 0.5065760450902114, 0.47118893985010646, 0.4801940897025039, 0.47715158308172767, 0.4759258811679655, 0.4747902206495273, 0.48807398516085054, 0.47823475416673916, 0.48244272193554716, 0.4589791905900623, 0.48145672836458997, 0.48879197185395484, 0.4780083851389135, 0.48198035406553263, 0.492791353997726, 0.4823964808155398, 0.49671421162029705]
y[8] = [0.3664153427487872, 0.39756716084580046, 0.4325665097453045, 0.408148728148558, 0.431346681417787, 0.43325818140386174, 0.4610444662941277, 0.4361965072985991, 0.46961157697228495, 0.4387633981392119, 0.4294561585665906, 0.4391840556873457, 0.4373299641423101, 0.4585544389810657, 0.45057044700295756, 0.4745244394460652, 0.4793634248034899, 0.45160117567592944, 0.47267419406884903, 0.4616466866913332, 0.4682483773875845, 0.4639019136576543, 0.458219125643765, 0.4434212248415474, 0.4668665872617478, 0.4645165518534148, 0.48545676679613037, 0.46598149132692684, 0.4612788012549426, 0.48292040647678913, 0.4846078394199759]
y[9] = [0.45399824440102576, 0.469114706642611, 0.48597580611997915, 0.487718344971308, 0.4828894858267405, 0.4828925812958882, 0.49420235145048325, 0.49204432091832695, 0.4897976839778644, 0.48132838041427634, 0.49685078023259427, 0.4750220983423637, 0.4778925478091562, 0.47831312100031625, 0.49136189893233495, 0.5047190467162141, 0.49001281794203944, 0.49567182043867736, 0.4877021721939109, 0.4909628756423234, 0.5098024189871408, 0.5008119432457726, 0.5169970703399482, 0.49988569718477865, 0.4983907434937259, 0.5086951480247615, 0.5133526957248867, 0.5158502566154899, 0.4990244847939372, 0.4975770312692275, 0.5119843927852713]
y[10] = [0.4553595589599076, 0.4479118695128689, 0.452445552389589, 0.45275733915525596, 0.42822004313950707, 0.4426152962043293, 0.4524848801326524, 0.44772776082984017, 0.4605113723697497, 0.4777446742920793, 0.45540135658760755, 0.47375032260915767, 0.47325158181938265, 0.4656209107994211, 0.46739560088746546, 0.4908580518161662, 0.47140429456827443, 0.4940855339746592, 0.4613331577385446, 0.4970074748997023, 0.47452370768350116, 0.4684440181882148, 0.5082270788727692, 0.4905661976027143, 0.4945846066840627, 0.501748623887272, 0.5024453373459107, 0.5004022913219838, 0.516908944619414, 0.5147959346340706, 0.48349438705700215]

f1 = [0.4239110986778336, 0.4652586810683659, 0.44674609503428847, 0.43710565877595053, 0.43716523046583466, 0.41843573202769574, 0.4779319157224645, 0.4567740396028196, 0.45340789932484327, 0.4322107735241118]
f2 = [0.4975577915147877, 0.48533547168712377, 0.48505116098496676, 0.4760733514301514, 0.4861523189155812, 0.48428246096541083, 0.48447903721145763, 0.5153238052830977, 0.4810575141662639, 0.4653224173815868]
f3 = [0.5133808989023838, 0.5035828592227588, 0.5107918062706917, 0.5024690327540648, 0.4846795542656506, 0.5046407750953981, 0.5187432530939766, 0.47661755749305496, 0.500102020931423, 0.4933596218983858]
xf = [2000, 3500, 5000]

x = [2000 + i*100 for i in range(31)]

iy = {}
iy[1] = [0.4320375925073071, 0.47456733794260736, 0.47326206685517047, 0.5003967258551937, 0.4907585697712138, 0.5101654096944214, 0.5175340916813259]
iy[2] = [0.44832874488745594, 0.4936533545214269, 0.4727750982137212, 0.4907318709644318, 0.49632689361043464, 0.5268102004247691, 0.5266277496442623]
iy[3] = [0.43538406694524345, 0.4592525092582198, 0.4681907127833396, 0.4968461297577017, 0.4904513146169358, 0.5229844092434256, 0.5012337477486456]
iy[4] = [0.46617646266265356, 0.4550679829135722, 0.48325765815413024, 0.4872593702127199, 0.494598153251665, 0.49550464794133786, 0.5137422051370012]
iy[5] = [0.4327911614937237, 0.46172661800175574, 0.4827658875322627, 0.49192798296892504, 0.4992879416078527, 0.5121251576846708, 0.5289303026154719]
iy[6] = [0.47199532790446874, 0.46945320539852203, 0.505175572925735, 0.5090989735290503, 0.5090875601872378, 0.5194632858409279, 0.5196362647799039]
iy[7] = [0.4564404755157271, 0.4679042665132312, 0.4962992131887301, 0.5053083793913071, 0.5155037085775377, 0.5354878500120507, 0.5172798920100871]
iy[8] = [0.4477966217616094, 0.4884574184864683, 0.4958512114765577, 0.5084469682191903, 0.508192767470639, 0.5162060326323907, 0.5221810882930552]
iy[9] = [0.47135442554256723, 0.4866814485019881, 0.49756505996567324, 0.5161781907512741, 0.5365067987900809, 0.526798323365161, 0.5304784698627553]
iy[10] = [0.4348352357432831, 0.4747549049069232, 0.49025914941891724, 0.4703080773095212, 0.49520769124487235, 0.501796808807113, 0.5104384356331811]

ix = [2000 + i*500 for i in range(7)]

ty5ok = {}
ty5ok[1] = [0.438, 0.446, 0.474, 0.466, 0.474, 0.471, 0.483, 0.493, 0.515, 0.487, 0.477, 0.499, 0.516, 0.497, 0.509, 0.504, 0.505,
 0.498, 0.506, 0.492, 0.519, 0.524, 0.501, 0.508, 0.515, 0.532, 0.535, 0.531, 0.533, 0.540, 0.531]
ty5ok[2] = [0.450, 0.414, 0.469, 0.481, 0.470, 0.469, 0.497, 0.481, 0.510, 0.510, 0.512, 0.513, 0.528, 0.534, 0.523, 0.515, 0.529,
 0.529, 0.517, 0.532, 0.532, 0.524, 0.530, 0.528, 0.539, 0.522, 0.548, 0.536, 0.537, 0.535, 0.553]


# for i in range(1, 11):
#     plt.plot(ix, iy[i], linewidth=0.5)


# for i in range(1, 11):
#     plt.plot(x, y[i], linewidth=0.5)

for i in range(1, 3):
    plt.plot(x, ty5ok[i], linewidth=0.5)

m = []
for i in range(31):
    b = [y[k][i] for k in range(1, 11)]
    m.append(mean(b))


im = []
for i in range(7):
    ib = [iy[k][i] for k in range(1, 11)]
    im.append(mean(ib))

jm = []
for i in range(31):
    ib = [ty5ok[k][i] for k in range(1, 3)]
    jm.append(mean(ib))


plt.plot(x, m, linewidth=4)
plt.plot(ix, im, linewidth=4)
plt.plot(x, jm, linewidth=4)

plt.scatter(xf[0], mean(f1), marker='+', c='r', s=150, linewidth=3)
plt.scatter(xf[1], mean(f2), marker='+', c='r', s=150, linewidth=3)
plt.scatter(xf[2], mean(f3), marker='+', c='r', s=150, linewidth=3)
# plt.scatter(x_al_2000_2, y_al_2000_2, marker='*', c='y', s=150)

# plt.xscale('log')
plt.grid(True)
plt.show()
