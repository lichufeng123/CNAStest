# 🐍 Python 项目协作开发指南

欢迎加入本项目！本仓库采用 Git 协同开发模式，以下为详细开发流程和规范。

---

## 📂 分支策略

| 分支               | 说明                                   |
| ------------------ | -------------------------------------- |
| `main`             | 主分支：仅存放稳定、可发布的版本       |
| `dev`              | 开发分支：所有功能合并至此，集成测试用 |
| `yourname-feature` | 功能分支：每人每个功能使用独立分支开发 |

---

## 🔧 开发流程（开发者日常操作）

### ✅ 第一次参与项目（仅需一次）

```bash
git clone https://github.com/yourname/project.git
cd project
git checkout dev
```

### 🔁 每次开发流程

1. 拉取最新 `dev` 分支代码：

```
git checkout dev
git pull origin dev
```

1. 创建并切换到自己的功能分支：

```

git checkout -b yourname-feature1
```

1. 编写代码，提交修改：

```
git add .
git commit -m "feat: 添加用户登录功能"
```

1. 推送功能分支到远程仓库：

```
git push -u origin yourname-feature1
```

1. 发起合并请求（Pull Request）至 `dev` 分支：
   - 打开 GitHub → 本项目 → 点击 `Compare & pull request`
   - 目标分支选择 `dev`，填写说明并提交 PR
   - 审核通过后由他人合并（避免自己合并自己的代码）

------

## ✏️ 提交规范

提交信息请使用以下格式：

```
<类型>: <简要描述>
```

| 类型     | 含义            |
| -------- | --------------- |
| feat     | 新功能          |
| fix      | 修复 Bug        |
| docs     | 文档变更        |
| style    | 格式变动        |
| refactor | 代码重构        |
| test     | 测试相关        |
| chore    | 构建/工具等变更 |



📌 **示例：**

```
git commit -m "fix: 修复登录接口密码校验逻辑"
```

------

## 🔀 合并流程

### 🔁 功能分支 → dev

```
# 确保在 dev 分支并更新
git checkout dev
git pull origin dev

# 合并你的功能分支
git merge yourname-feature1

# 推送 dev 到远程
git push origin dev
```

### 🚀 最终发布：dev → main（由负责人操作）

```
git checkout main
git pull origin main
git merge dev
git push origin main
```

------

## 🧹 常见问题

| 问题       | 解决办法                                      |
| ---------- | --------------------------------------------- |
| 看不到分支 | 使用 `git push -u origin 分支名` 推送后可见   |
| 合并冲突   | `git pull --rebase` 后解决冲突再提交          |
| 推送被拒   | 确保本地为最新，可 `git pull` 同步后再 `push` |