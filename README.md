# InstantSplat
An unofficial implementation of InstantSplat, an sparse-view, SfM-free framework for large-scale scene reconstruction method using Gaussian Splatting.
Uses [Rerun](https://rerun.io/) to visualize, [Gradio](https://www.gradio.app) for an interactive UI, and [Pixi](https://pixi.sh/latest/) for a easy installation

<p align="center">
    <a title="Website" href="https://instantsplat.github.io/" target="_blank" rel="noopener noreferrer" style="display: inline-block;">
        <img src="https://www.obukhov.ai/img/badges/badge-website.svg">
    </a>
    <a href='https://huggingface.co/spaces/pablovela5620/instant-splat'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue'></a>
    <a title="arXiv" href="https://arxiv.org/abs/2403.20309" target="_blank" rel="noopener noreferrer" style="display: inline-block;">
        <img src="https://www.obukhov.ai/img/badges/badge-pdf.svg">
    </a>
    <a title="Github" href="https://github.com/pablovela5620/InstantSplat" target="_blank" rel="noopener noreferrer" style="display: inline-block;">
        <img src="https://img.shields.io/github/stars/pablovela5620/InstantSplat?label=GitHub%20%E2%98%85&logo=github&color=C8C" alt="badge-github-stars">
    </a>
    <a title="Social" href="https://x.com/pablovelagomez1" target="_blank" rel="noopener noreferrer" style="display: inline-block;">
        <img src="https://www.obukhov.ai/img/badges/badge-social.svg" alt="social">
    </a>
  </p>

<p align="center">
  <img src="media/final_instantsplat.gif" alt="example output" width="720" />
</p>

## Install and Run
Make sure you have the [Pixi](https://pixi.sh/latest/#installation) package manager installed
```bash
git clone https://github.com/pablovela5620/InstantSplat.git
cd InstantSplat
pixi run app
```

All commands can be listed using `pixi task list`
## Hosted Demo
Demos can be found on huggingface spaces, local version is recommended to avoid GPU timeouts!

<a href='https://huggingface.co/spaces/pablovela5620/instant-splat'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue'></a>

## Acknowledgements
Thanks to the original InstantSplat, Gaussian Splatting, and Dust3r repos!

[InstantSplat](https://github.com/NVlabs/InstantSplat)
```bibtex
@misc{fan2024instantsplat,
        title={InstantSplat: Unbounded Sparse-view Pose-free Gaussian Splatting in 40 Seconds},
        author={Zhiwen Fan and Wenyan Cong and Kairun Wen and Kevin Wang and Jian Zhang and Xinghao Ding and Danfei Xu and Boris Ivanovic and Marco Pavone and Georgios Pavlakos and Zhangyang Wang and Yue Wang},
        year={2024},
        eprint={2403.20309},
        archivePrefix={arXiv},
        primaryClass={cs.CV}
      }
```
[Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting)
```bibtex
@Article{kerbl3Dgaussians,
      author       = {Kerbl, Bernhard and Kopanas, Georgios and Leimk{\"u}hler, Thomas and Drettakis, George},
      title        = {3D Gaussian Splatting for Real-Time Radiance Field Rendering},
      journal      = {ACM Transactions on Graphics},
      number       = {4},
      volume       = {42},
      month        = {July},
      year         = {2023},
      url          = {https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/}
}
```
[Dust3r](https://github.com/naver/dust3r)
```bibtex
@inproceedings{dust3r_cvpr24,
      title={DUSt3R: Geometric 3D Vision Made Easy}, 
      author={Shuzhe Wang and Vincent Leroy and Yohann Cabon and Boris Chidlovskii and Jerome Revaud},
      booktitle = {CVPR},
      year = {2024}
}

@misc{dust3r_arxiv23,
      title={DUSt3R: Geometric 3D Vision Made Easy}, 
      author={Shuzhe Wang and Vincent Leroy and Yohann Cabon and Boris Chidlovskii and Jerome Revaud},
      year={2023},
      eprint={2312.14132},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```