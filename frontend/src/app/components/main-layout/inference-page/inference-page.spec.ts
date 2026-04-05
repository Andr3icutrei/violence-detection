import { ComponentFixture, TestBed } from '@angular/core/testing';

import { InferencePage } from './inference-page';

describe('InferencePage', () => {
  let component: InferencePage;
  let fixture: ComponentFixture<InferencePage>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [InferencePage],
    }).compileComponents();

    fixture = TestBed.createComponent(InferencePage);
    component = fixture.componentInstance;
    await fixture.whenStable();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
