### hexo博客安装步骤
##### 安装git 和node.js
省略掉。我这里安装的版本为
```
$ git --version && node -v
git version 2.19.1.windows.1
v10.13.0
```
#### 安装hexo
```
npm install hexo -g
```
安装完成之后，输入hexo -v 检查hexo是否安装成功。

#### 启动项目
有两种方式，一种直接新建项目，一种，拿现在仓库项目直接跑起来。
- 新建空的项目
    输入 hexo init,初始化文件夹
    ```
    hexo init
    npm install
    ```
    生成页面
    ```
    hexo g
    ```
    启动服务
    ```
    hexo s
    ```
- 直接使用仓库代码创建
    ```
    git clone git@github.com:hobbitmr/hexoblog.git
    npm install
    ```


