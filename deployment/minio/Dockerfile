FROM golang:1.24-alpine AS build

ARG MINIO_COMMIT=07c3a429bfed433e49018cb0f78a52145d4bedeb
ENV CGO_ENABLED=0
WORKDIR /src
ADD https://github.com/minio/minio/archive/${MINIO_COMMIT}.tar.gz /tmp/minio-source.tar.gz
RUN tar --extract --gzip --file=/tmp/minio-source.tar.gz --strip-components=1 --directory=/src \
    && go build -tags kqueue -trimpath -o /out/minio .

FROM alpine:3.23

RUN apk add --no-cache ca-certificates curl \
    && addgroup -S -g 10001 minio \
    && adduser -S -D -H -u 10001 -G minio minio \
    && mkdir -p /data \
    && chown minio:minio /data
COPY --from=build /out/minio /usr/local/bin/minio

USER 10001:10001
EXPOSE 9000 9001
VOLUME ["/data"]
ENTRYPOINT ["/usr/local/bin/minio"]
CMD ["server", "/data", "--console-address", ":9001"]
