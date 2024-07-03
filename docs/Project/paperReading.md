### A Deep Dive into Single-Cell RNA Sequencing Foundation Models

- 比较scBERT 和 scGPT 并用cell type annotation和fine-tuning 来比较两个模型 （+logistic regression对比）
- 介绍Large-scale foundation models

### Blood protein levels predict leading incident diseases and mortality in UK Biobank

- 利用Blood protein levels去预测疾病
- We report 3,201 associations between 961 protein levels and 21 incident outcomes, identifying proteomic indicators of multiple morbidities.
- 介绍了Biobank数据和Biobank protein 数据

### Genetic regulation of the human plasma proteome in 54,306 UK Biobank participants

- 介绍Biobank数据

### Genetics meets proteomics: perspectives for large population-based studies

- In this Review, we focus on large-scale proteomic technologies currently capable of profiling the blood circulating proteome in large population studies.
- The human plasma proteome 人血浆蛋白质组
- Experimental coverage of the human proteome 人类蛋白质组的实验覆盖
- The plasma proteome 血浆蛋白质组
    - Proteins circulating in plasma can theoretically origi- nate from any organ or cell, or can even pass through the placenta to be exchanged between mother and child (血浆中循环的蛋白质理论上可以源自任何器官或细胞，甚至可以通过胎盘在母婴之间进行交换)
    - the presence of cellular proteins in the bloodstream may be a conse- quence of a variety of different and complex processes, including response to damage to cells and organs (血液中细胞蛋白的存在可能是各种不同和复杂过程的结果，包括对细胞和器官损伤的反应)
- Genetic variance in clinical biomarker proteins (临床生物标志物蛋白质的遗传变异)
    - The measurement of protein levels in accessible biofluids (for example, plasma, urine or cerebrospinal fluid) is one of the mainstays of clinical medicine, providing many biomarkers with diagnostic or prognostic value. (测量可获取的生物液体（例如血浆、尿液或脑脊液）中的蛋白质水平是临床医学的支柱之一，为许多生物标志物提供诊断或预后价值)

结论与展望

利用中间表型的遗传关联研究代表了由自然进行的实验分析，能够提供宝贵的生物学信息资源。规模化的蛋白质组数据最近才对大规模全基因组关联研究（GWAS）变得可访问，这在很大程度上是由于与血浆蛋白质组成相关的分析限制。亲和蛋白质组学技术的进步正在迅速弥补这一不足，最近的研究报道了在近17,000个血液样本中同时测量5,000种蛋白质，许多更大规模的研究正在进行中。目前，发现的所有蛋白质定量特征位点（pQTLs）中有10%到20%与临床GWAS位点共定位。随着pQTL研究和临床GWAS的规模和范围的持续扩大，这种重叠预计会增加。这将增加血浆蛋白质组数据的价值，特别是对生物医学和制药领域，通过提供更多更好的工具进行药物靶点验证和因果推断。重要的是，大规模的pQTL研究在大型健康队列中补充了在病例对照数据集和临床试验中日益使用的血浆蛋白质组数据的使用，改善了受逆向因果和混杂影响的研究设计中的推断。

尽管亲和性测定在pQTL定位中领先，但明智的做法是要意识到它们的限制。感兴趣的关联应该通过独立验证靶标特异性得到支持，并应考虑可能的交叉反应和表位效应。由于目前没有一种方法是最优的或唯一的解决方案，因此在多种技术上进行验证将是关键，最重要的发现还应该通过细胞和其他功能研究进一步支持。大规模GWAS项目中基于质谱（MS）方法的可行性尚待证明。尽管低样本通量和低分析敏感性目前限制了MS技术在人群研究中的使用，MS具有识别更广泛的肽和蛋白质范围的潜力，以及各种翻译后修饰和剪接形式。已经报道了与IgG糖基化和总N-糖链有生物学相关性的关联。其他方法，称为“糖蛋白质组学”，旨在理解血浆蛋白质标志物的差异性糖基化。因此，未来的努力应该尝试通过高通量和定量技术捕捉蛋白质的不同修饰特性，并可能确定与这些特性相关的组织特异性异构体。同时，将大规模蛋白质组数据与如DNA甲基化、代谢组学和糖组学等分子读数结合的前景，为在人群规模研究中从分子层面研究人类健康开辟了新途径。

近期在血浆和其他生物流体的蛋白质组内容调查的规模和范围上取得的进展，使蛋白质组学能够与其他组学方法（如聚焦于遗传变异和RNA表达的方法）一道，进行全面的特征描述。这些进展为使用蛋白质组学提供了改善疾病机制基础理解的新机会，并通过靶标和生物标志物识别促进新的转化策略。

### Organ aging signatures in the plasma proteome track health and disease

- We utilized levels of human blood plasma proteins originating from specific organs to measure organ-specific aging differences in living individuals.
- 利用血浆蛋白质组学数据研究器官衰老，预测疾病和衰老影响

### Plasma proteomic associations with genetics and health in the UK Biobank

- 提供了该计划的详细总结，包括技术和生物学验证、对蛋白质组疾病特征的见解以及各种人口和健康指标的预测模型


### Transfer learning enables predictions in network biology

- 开发了一种上下文感知、基于注意力的深度学习模型 Geneformer，该模型在约 3000 万个单细胞转录组的大规模语料库上进行了预训练，以便在网络生物学数据有限的环境中实现上下文特定的预测
- Geneformer 获得了对网络动态、编码网络层次结构的基本了解以完全自我监督的方式调整模型的注意力权重
- Geneformer 代表了一种预训练的深度学习模型，可以对广泛的下游应用进行微调，以加速关键网络调节因子和候选治疗靶点的发现