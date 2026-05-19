import { Component, EventEmitter, Input, Output } from '@angular/core';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-paginator',
  imports: [CommonModule],
  templateUrl: './paginator.html',
  styleUrl: './paginator.css',
  standalone: true,
})
export class Paginator {
  @Input() page: number = 0;
  @Input() pageSize: number = 40;
  @Input() hasMore: boolean = true;
  @Input() totalPages?: number;
  @Output() pageChange = new EventEmitter<number>();

  get displayPages(): number[] {
    const pages: number[] = [];
    const current = this.page;
    pages.push(current);
    if (this.hasMore) {
      pages.push(current + 1);
    }
    if (current > 0) {
      pages.unshift(current - 1);
    }
    return pages;
  }

  goToPage(p: number): void {
    if (p !== this.page && p >= 0) {
      this.pageChange.emit(p);
    }
  }

  firstPage(): void {
    if (this.page > 0) {
      this.pageChange.emit(0);
    }
  }

  previousPage(): void {
    if (this.page > 0) {
      this.pageChange.emit(this.page - 1);
    }
  }

  nextPage(): void {
    if (this.hasMore) {
      this.pageChange.emit(this.page + 1);
    }
  }

  lastPage(): void {
    if (this.totalPages !== undefined && this.page < this.totalPages - 1) {
      this.pageChange.emit(this.totalPages - 1);
    }
  }
}
