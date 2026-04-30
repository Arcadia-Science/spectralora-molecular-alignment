FROM node:20-alpine AS build
WORKDIR /app
COPY apps/frontend/package.json ./
RUN npm install
COPY apps/frontend/index.html .
COPY apps/frontend/vite.config.js .
COPY apps/frontend/postcss.config.js .
COPY apps/frontend/tailwind.config.js .
COPY apps/frontend/src ./src
ARG VITE_API_URL=http://localhost:8000
ENV VITE_API_URL=$VITE_API_URL
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
